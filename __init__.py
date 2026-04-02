import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from bpy_extras import view3d_utils
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import bmesh
import math
import time
import numpy as np


# ---------------------------------------------------------------------------
# Cached shaders
# ---------------------------------------------------------------------------

_line_shader = None
_point_shader = None


def _get_line_shader():
    global _line_shader
    if _line_shader is None:
        _line_shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    return _line_shader


def _get_point_shader():
    global _point_shader
    if _point_shader is None:
        _point_shader = gpu.shader.from_builtin('POINT_UNIFORM_COLOR')
    return _point_shader


# ---------------------------------------------------------------------------
# Draw callback
# ---------------------------------------------------------------------------

CLOSE_RADIUS = 12


def draw_callback_px(state, context):
    # Draw state is stored in a plain dict so the callback stays safe even if
    # the operator RNA is already freed by Blender.
    poly_points = state.get("poly_points")
    if not poly_points:
        return

    shader = _get_line_shader()
    vp = (context.region.width, context.region.height)
    mx, my = state.get("mouse_pos", (0, 0))

    gpu.state.blend_set('ALPHA')
    shader.uniform_float("viewportSize", vp)
    shader.uniform_float("lineWidth", 1.5)

    # --- Placed segments + rubber band (white) ---
    coords = list(poly_points) + [(mx, my)]
    shader.uniform_float("color", (1.0, 1.0, 1.0, 0.9))
    batch_for_shader(shader, 'LINE_STRIP', {"pos": coords}).draw(shader)

    # --- Closing edge back to start (blue) ---
    if len(poly_points) >= 2:
        shader.uniform_float("color", (0.3, 0.5, 1.0, 0.6))
        closing = [(mx, my), poly_points[0]]
        batch_for_shader(shader, 'LINE_STRIP', {"pos": closing}).draw(shader)

    # --- Start point marker ---
    pt_shader = _get_point_shader()
    sx, sy = poly_points[0]
    near = math.hypot(mx - sx, my - sy) <= CLOSE_RADIUS and len(poly_points) >= 3
    gpu.state.point_size_set(10.0 if near else 6.0)
    pt_shader.uniform_float("color", (1.0, 1.0, 1.0, 1.0 if near else 0.7))
    start_batch = batch_for_shader(pt_shader, 'POINTS', {"pos": [(sx, sy)]})
    start_batch.draw(pt_shader)

    gpu.state.point_size_set(1.0)
    gpu.state.blend_set('NONE')


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

class POLY_OT_select(bpy.types.Operator):
    """Draw a polygon to select objects or mesh elements"""
    bl_idname = "view3d.poly_select"
    bl_label = "Polygonal Lasso Select"
    bl_options = {'REGISTER', 'UNDO'}

    # ---- point-in-polygon -------------------------------------------------

    def _point_in_poly(self, px, py):
        crossings = 0
        pts = self._draw_state["poly_points"]
        n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            if (y1 > py) != (y2 > py):
                xint = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
                if px < xint:
                    crossings += 1
        return crossings % 2 != 0

    # ---- occlusion -------------------------------------------------------

    def _is_xray_or_wireframe(self, context):
        shading = context.space_data.shading
        if shading.type == 'WIREFRAME':
            return True
        if shading.show_xray:
            return True
        return False

    # ---- helpers ----------------------------------------------------------

    def _get_viewport(self, context):
        for region in context.area.regions:
            if region.type == 'WINDOW':
                return region
        return None

    def _project(self, viewport, region_3d, point3d):
        pt = view3d_utils.location_3d_to_region_2d(viewport, region_3d, point3d)
        if pt is None:
            return None
        return (pt[0], pt[1])

    def _near_start(self):
        poly_points = self._draw_state["poly_points"]
        if len(poly_points) < 3:
            return False
        sx, sy = poly_points[0]
        mx, my = self._draw_state["mouse_pos"]
        return math.hypot(mx - sx, my - sy) <= CLOSE_RADIUS

    # ---- batch point-in-polygon (numpy) -----------------------------------

    def _points_in_poly_np(self, points_2d):
        """Vectorised ray-cast for an Nx2 numpy array. Returns bool array."""
        px = points_2d[:, 0]
        py = points_2d[:, 1]
        poly = np.array(self._draw_state["poly_points"], dtype=np.float64)
        n = len(poly)
        inside = np.zeros(len(px), dtype=np.bool_)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            # Which points have the ray crossing this edge?
            cond = (y1 > py) != (y2 > py)
            if not np.any(cond):
                continue
            xint = x1 + (py[cond] - y1) * (x2 - x1) / (y2 - y1)
            inside[cond] ^= (px[cond] < xint)
        return inside

    # ---- selection --------------------------------------------------------

    def _apply_selection(self, context, event):
        viewport = self._get_viewport(context)
        if viewport is None:
            return
        region_3d = context.area.spaces[0].region_3d

        extend = event.shift
        subtract = event.ctrl

        if not extend and not subtract:
            if context.mode == 'OBJECT':
                bpy.ops.object.select_all(action='DESELECT')
            elif context.mode == 'EDIT_MESH':
                bpy.ops.mesh.select_all(action='DESELECT')

        select_val = not subtract

        if context.mode == 'OBJECT':
            for obj in context.visible_objects:
                pt = self._project(viewport, region_3d, obj.location)
                if pt and self._point_in_poly(pt[0], pt[1]):
                    obj.select_set(select_val)

        elif context.mode == 'EDIT_MESH':
            activeobj = context.active_object
            bm = bmesh.from_edit_mesh(activeobj.data)
            bm.verts.ensure_lookup_table()
            mat = activeobj.matrix_world

            # Build the perspective projection matrix
            rv3d = region_3d
            w = viewport.width
            h = viewport.height
            proj_mat = rv3d.perspective_matrix  # 4x4: projection * view

            # Get all vert coords into numpy via mesh (faster than bmesh iteration)
            # Use bmesh vertex count, not mesh.vertices, to stay in sync with bm.verts indices
            vert_count = len(bm.verts)
            coords = np.empty(vert_count * 3, dtype=np.float64)
            for vi, v in enumerate(bm.verts):
                coords[vi * 3:(vi + 1) * 3] = v.co
            coords = coords.reshape(-1, 3)

            # To homogeneous (Nx4)
            ones = np.ones((vert_count, 1), dtype=np.float64)
            world = np.array(mat, dtype=np.float64)  # 4x4
            coords_h = np.hstack((coords, ones))  # Nx4

            # World transform then projection
            clip = (world @ coords_h.T).T       # Nx4 in world space
            pm = np.array(proj_mat, dtype=np.float64)
            ndc = (pm @ clip.T).T                # Nx4 in clip space

            # Perspective divide — skip verts behind camera
            w_clip = ndc[:, 3]
            valid = w_clip > 1e-6
            ndc_x = np.zeros(vert_count, dtype=np.float64)
            ndc_y = np.zeros(vert_count, dtype=np.float64)
            ndc_x[valid] = ndc[valid, 0] / w_clip[valid]
            ndc_y[valid] = ndc[valid, 1] / w_clip[valid]

            # NDC (-1..1) to region pixels
            screen_x = (ndc_x * 0.5 + 0.5) * w
            screen_y = (ndc_y * 0.5 + 0.5) * h

            pts_2d = np.column_stack((screen_x, screen_y))
            inside = self._points_in_poly_np(pts_2d) & valid

            # Occlusion check: hide verts behind faces unless x-ray/wireframe
            if not self._is_xray_or_wireframe(context):
                bvh = BVHTree.FromBMesh(bm)
                mat_inv = mat.inverted()

                if rv3d.is_perspective:
                    view_origin_local = mat_inv @ rv3d.view_matrix.inverted().translation
                    for i in np.where(inside)[0]:
                        v_co = bm.verts[i].co
                        direction = (v_co - view_origin_local).normalized()
                        vert_dist = (v_co - view_origin_local).length
                        hit = bvh.ray_cast(view_origin_local, direction)
                        if hit[0] is not None and hit[3] < vert_dist - 0.001:
                            inside[i] = False
                else:
                    # Orthographic: parallel rays along view direction
                    view_dir_local = (mat_inv.to_3x3()
                                      @ rv3d.view_matrix.inverted().to_3x3()
                                      @ Vector((0, 0, -1))).normalized()
                    for i in np.where(inside)[0]:
                        v_co = bm.verts[i].co
                        ray_origin = v_co - view_dir_local * 10000.0
                        hit = bvh.ray_cast(ray_origin, view_dir_local)
                        if hit[0] is not None and hit[3] < 10000.0 - 0.001:
                            inside[i] = False

            # Apply selection
            for i in np.where(inside)[0]:
                if 0 <= i < vert_count:
                    bm.verts[i].select = select_val

            bmesh.update_edit_mesh(activeobj.data)

    # ---- finish / cancel --------------------------------------------------

    def _remove_draw_handle(self):
        handle = getattr(self, "_handle", None)
        if handle is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
            except (ReferenceError, RuntimeError, ValueError):
                pass
            self._handle = None

    def _finish(self, context, event):
        self._apply_selection(context, event)
        self._remove_draw_handle()
        context.area.tag_redraw()
        return {'FINISHED'}

    def _cancel(self, context):
        self._remove_draw_handle()
        context.area.tag_redraw()
        return {'CANCELLED'}

    # ---- modal loop -------------------------------------------------------

    _handled_types = {'MOUSEMOVE', 'LEFTMOUSE', 'RIGHTMOUSE', 'ESC', 'BACK_SPACE'}
    _blocked_types = {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE',
                      'TRACKPADPAN', 'TRACKPADZOOM', 'NUMPAD_PERIOD',
                      'NDOF_MOTION'}

    def modal(self, context, event):
        # Block viewport navigation (orbit, pan, zoom) during selection
        if event.type in self._blocked_types:
            return {'RUNNING_MODAL'}

        if event.type not in self._handled_types:
            return {'PASS_THROUGH'}

        if event.type == 'MOUSEMOVE':
            self._draw_state["mouse_pos"] = [event.mouse_region_x, event.mouse_region_y]
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            # Close if clicking near start point
            if self._near_start():
                return self._finish(context, event)

            self._draw_state["poly_points"].append((event.mouse_region_x, event.mouse_region_y))

            # Double-click detection
            now = time.monotonic()
            if now - self._last_click_time < 0.3 and len(self._draw_state["poly_points"]) >= 3:
                return self._finish(context, event)
            self._last_click_time = now

        elif event.type == 'BACK_SPACE' and event.value == 'PRESS':
            # Undo last point
            if self._draw_state["poly_points"]:
                self._draw_state["poly_points"].pop()
            if not self._draw_state["poly_points"]:
                return self._cancel(context)

        elif event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
            return self._cancel(context)

        return {'RUNNING_MODAL'}

    # ---- invoke -----------------------------------------------------------

    def invoke(self, context, event):
        if context.area.type != 'VIEW_3D':
            self.report({'WARNING'}, "View3D not found, cannot run operator")
            return {'CANCELLED'}

        self._last_click_time = time.monotonic()
        self._draw_state = {
            "mouse_pos": [event.mouse_region_x, event.mouse_region_y],
            "poly_points": [(event.mouse_region_x, event.mouse_region_y)],
        }
        self._handle = None

        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            draw_callback_px, (self._draw_state, context), 'WINDOW', 'POST_PIXEL'
        )
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}


# ---------------------------------------------------------------------------
# Toolbar tool
# ---------------------------------------------------------------------------

class POLY_TL_select(bpy.types.WorkSpaceTool):
    bl_space_type = 'VIEW_3D'
    bl_context_mode = 'OBJECT'
    bl_idname = "view3d.poly_lasso_select"
    bl_label = "Poly Lasso Select"
    bl_description = "Select objects by drawing a polygon"
    bl_icon = "ops.generic.select_lasso"
    bl_widget = None
    bl_keymap = (
        ("view3d.poly_select", {"type": 'LEFTMOUSE', "value": 'PRESS'}, None),
    )


class POLY_TL_select_edit(bpy.types.WorkSpaceTool):
    bl_space_type = 'VIEW_3D'
    bl_context_mode = 'EDIT_MESH'
    bl_idname = "view3d.poly_lasso_select_edit"
    bl_label = "Poly Lasso Select"
    bl_description = "Select vertices by drawing a polygon"
    bl_icon = "ops.generic.select_lasso"
    bl_widget = None
    bl_keymap = (
        ("view3d.poly_select", {"type": 'LEFTMOUSE', "value": 'PRESS'}, None),
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def draw_menu(self, context):
    self.layout.operator(POLY_OT_select.bl_idname, text="Polygonal Lasso Select")


def _select_menus():
    menus = []
    for name in ('VIEW3D_MT_select_object', 'VIEW3D_MT_select_edit_mesh',
                 'VIEW3D_MT_select', 'VIEW3D_MT_object', 'VIEW3D_MT_edit_mesh'):
        cls = getattr(bpy.types, name, None)
        if cls is not None:
            menus.append(cls)
    return menus


def register():
    bpy.utils.register_class(POLY_OT_select)
    bpy.utils.register_tool(POLY_TL_select, after={"builtin.select_lasso"}, separator=False)
    bpy.utils.register_tool(POLY_TL_select_edit, after={"builtin.select_lasso"}, separator=False)
    for menu in _select_menus():
        menu.append(draw_menu)


def unregister():
    for menu in _select_menus():
        menu.remove(draw_menu)
    bpy.utils.unregister_tool(POLY_TL_select)
    bpy.utils.unregister_tool(POLY_TL_select_edit)
    bpy.utils.unregister_class(POLY_OT_select)


if __name__ == "__main__":
    register()
