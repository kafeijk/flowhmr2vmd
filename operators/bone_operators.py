"""Bone-level operators: standardize channels and FK to MMD IK conversion."""
import bpy
from mathutils import Matrix, Quaternion, Vector

from ..core import mmd

_BONE_PROGRESS = {
    'active': False,
    'factor': 0.0,
    'text': '',
}


def get_bone_progress():
    return _BONE_PROGRESS


def set_bone_progress(active, factor=0.0, text=''):
    _BONE_PROGRESS['active'] = bool(active)
    _BONE_PROGRESS['factor'] = min(max(float(factor), 0.0), 1.0)
    _BONE_PROGRESS['text'] = text


def tag_redraw(context):
    window = getattr(context, 'window', None)
    if window is None or window.screen is None:
        return
    for area in window.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()


def _remove_upper_body_children(context, armature_obj, root_name='上半身'):
    bone = armature_obj.data.bones.get(root_name)
    if bone is None:
        return
    names = []
    stack = list(bone.children)
    while stack:
        child = stack.pop()
        names.append(child.name)
        stack.extend(child.children)
    for obj in list(context.selected_objects):
        obj.select_set(False)
    armature_obj.select_set(True)
    context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = armature_obj.data.edit_bones
    for name in names:
        edit_bone = edit_bones.get(name)
        if edit_bone is not None:
            edit_bones.remove(edit_bone)
    bpy.ops.object.mode_set(mode='OBJECT')


def _set_leg_ik(armature, enabled):
    # 足首 的 IK（target つま先ＩＫ, chain 1）决定脚掌朝向，必须一起开关，
    # 否则足ＩＫ 的旋转对脚不起作用。
    for side in ('.L', '.R'):
        for name in (f'ひざ{side}', f'足首{side}'):
            bone = armature.pose.bones.get(name)
            if bone is None:
                continue
            for con in bone.constraints:
                if con.type == 'IK':
                    con.mute = not enabled
                    if enabled:
                        con.influence = 1.0


_REST_CACHE = {}


def _rest_matrices(armature):
    """足ＩＫ / つま先ＩＫ 的世界静息朝向（按骨架数据名缓存）。"""
    key = armature.data.name
    cached = _REST_CACHE.get(key)
    if cached is None:
        cached = {}
        for name in ('足ＩＫ.L', '足ＩＫ.R', 'つま先ＩＫ.L', 'つま先ＩＫ.R'):
            bone = armature.data.bones.get(name)
            if bone is not None:
                cached[name] = (armature.matrix_world @
                                bone.matrix_local).to_quaternion().to_matrix().to_4x4()
        _REST_CACHE[key] = cached
    return cached


def _convert_frame(armature, scene, frame, records):
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    targets = {}
    for side in ('.L', '.R'):
        ankle = armature.pose.bones.get(f'足首{side}')
        ik = armature.pose.bones.get(f'足ＩＫ{side}')
        toe = armature.pose.bones.get(f'つま先ＩＫ{side}')
        if ankle is None or ik is None or toe is None:
            continue
        # 足ＩＫ 只负责把脚踝带到原位（位置），旋转保持自身静息朝向；
        # 脚掌朝向导交给 つま先ＩＫ + 足首 的 IK，这样足ＩＫ 不需要补偿
        # 与足首的静息差，也不会出现 head/tail 强行共线的大角度。
        targets[side] = ((armature.matrix_world @ ankle.matrix).translation.copy(),
                         (armature.matrix_world @ ankle.tail).copy())
    if not targets:
        return
    world_inv = armature.matrix_world.inverted()
    rest = _rest_matrices(armature)
    for side, (head, tail) in targets.items():
        ik = armature.pose.bones.get(f'足ＩＫ{side}')
        ik.rotation_mode = 'QUATERNION'
        ik.matrix = world_inv @ (Matrix.Translation(head) @ rest[f'足ＩＫ{side}'])
        bpy.context.view_layer.update()
        toe = armature.pose.bones.get(f'つま先ＩＫ{side}')
        toe.rotation_mode = 'QUATERNION'
        toe.matrix = world_inv @ (Matrix.Translation(tail) @ rest[f'つま先ＩＫ{side}'])
        records[side].append((frame,
                              ik.location.copy(),
                              ik.rotation_quaternion.copy(),
                              bool(ik.mmd_ik_toggle),
                              toe.location.copy()))
    bpy.context.view_layer.update()


def _write_back_leg_ik(armature, records):
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        return
    for side, rows in records.items():
        if not rows:
            continue
        frames = [row[0] for row in rows]
        specs = (
            (f'pose.bones["足ＩＫ{side}"].location', 3, lambda row: row[1]),
            (f'pose.bones["足ＩＫ{side}"].rotation_quaternion', 4,
             lambda row: row[2]),
            (f'pose.bones["足ＩＫ{side}"].mmd_ik_toggle', 1,
             lambda row: (row[3],)),
            (f'pose.bones["つま先ＩＫ{side}"].location', 3, lambda row: row[4]),
        )
        for path, count, getter in specs:
            for fcurve in [fc for fc in action.fcurves
                           if fc.data_path == path]:
                action.fcurves.remove(fcurve)
            for index in range(count):
                fcurve = action.fcurves.new(path, index=index,
                                            action_group='fk_ik')
                fcurve.keyframe_points.add(len(frames))
                points = fcurve.keyframe_points
                for i, row in enumerate(rows):
                    value = getter(row)
                    point = points[i]
                    point.co = (frames[i], float(value[index]))
                    point.interpolation = 'BEZIER'
                fcurve.update()


def _remove_leg_fcurves(armature):
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        return
    rotation_paths = ('rotation_quaternion', 'rotation_euler',
                      'rotation_axis_angle')
    for fcurve in list(action.fcurves):
        path = fcurve.data_path
        for side in ('.L', '.R'):
            for name in (f'ひざ{side}', f'足首{side}', f'足先EX{side}'):
                if not path.startswith(f'pose.bones["{name}"]'):
                    continue
                if path.endswith(rotation_paths):
                    continue
                action.fcurves.remove(fcurve)
                break


_CENTER_BONE = 'センター'
_WAIST_BONE = '腰'
_MASTER_BONE = '全ての親'


def _curves(action, path, index=None):
    return [fc for fc in action.fcurves
            if fc.data_path == path and
            (index is None or fc.array_index == index)]


def _curve_map(action):
    return {(fc.data_path, fc.array_index): fc for fc in action.fcurves}


def _key_times(action, path):
    times = set()
    for fcurve in _curves(action, path):
        times.update(point.co[0] for point in fcurve.keyframe_points)
    return times


def _evaluate(curves, path, index, frame, default=0.0):
    fcurve = curves.get((path, index))
    if fcurve is None:
        return default
    return fcurve.evaluate(frame)


def _quaternion_at(curves, path, frame):
    values = [_evaluate(curves, path, index, frame, 1.0 if index == 0 else 0.0)
              for index in range(4)]
    return Quaternion(values).normalized()


def _remove_curves(action, path):
    for fcurve in _curves(action, path):
        action.fcurves.remove(fcurve)


def _write_curves(action, path, frames, columns, group='standardize'):
    _remove_curves(action, path)
    for index, values in enumerate(columns):
        fcurve = action.fcurves.new(path, index=index, action_group=group)
        fcurve.keyframe_points.add(len(frames))
        points = fcurve.keyframe_points
        for i, frame in enumerate(frames):
            point = points[i]
            point.co = (frame, float(values[i]))
            point.interpolation = 'LINEAR'
        fcurve.update()


def _move_standardize_channels(action, armature):
    master_path = f'pose.bones["{_MASTER_BONE}"].location'
    master_rotation_path = f'pose.bones["{_MASTER_BONE}"].rotation_quaternion'
    waist_path = f'pose.bones["{_WAIST_BONE}"].rotation_quaternion'
    center_location_path = f'pose.bones["{_CENTER_BONE}"].location'
    center_rotation_path = f'pose.bones["{_CENTER_BONE}"].rotation_quaternion'
    has_location = bool(_curves(action, master_path))
    has_rotation = bool(_curves(action, waist_path))
    if not has_location and not has_rotation:
        return False
    ik_targets = []
    if has_location:
        for side in ('.L', '.R'):
            name = f'足ＩＫ{side}'
            bone = armature.data.bones.get(name)
            if bone is None or bone.parent is None:
                continue
            if _curves(action, f'pose.bones["{name}"].location'):
                ik_targets.append(name)
    frames = set()
    for path in (master_path, master_rotation_path, waist_path,
                 center_location_path, center_rotation_path):
        frames |= _key_times(action, path)
    for name in ik_targets:
        frames |= _key_times(action, f'pose.bones["{name}"].location')
    frames = sorted(frames)
    if not frames:
        return False
    curves = _curve_map(action)
    rest_master = armature.data.bones[_MASTER_BONE].matrix_local.copy()
    rest_center = armature.data.bones[_CENTER_BONE].matrix_local.copy()
    rest_waist = armature.data.bones[_WAIST_BONE].matrix_local.copy()
    center_locations = ([], [], [])
    center_rotations = ([], [], [], [])
    ik_locations = {name: ([], [], []) for name in ik_targets}
    for frame in frames:
        master_location = Vector([_evaluate(curves, master_path, index, frame)
                                  for index in range(3)])
        master_rotation = _quaternion_at(curves, master_rotation_path, frame)
        waist_rotation = _quaternion_at(curves, waist_path, frame)
        center_basis = (
            Matrix.Translation([_evaluate(curves, center_location_path, index, frame)
                                for index in range(3)]) @
            _quaternion_at(curves, center_rotation_path, frame).to_matrix().to_4x4())
        master_basis = (Matrix.Translation(master_location) @
                        master_rotation.to_matrix().to_4x4())
        master_after = master_rotation.to_matrix().to_4x4()
        waist_pose = (rest_master @ master_basis @ rest_master.inverted() @
                      rest_center @ center_basis @ rest_center.inverted() @
                      rest_waist @ waist_rotation.to_matrix().to_4x4())
        target = (rest_master @ master_after @ rest_master.inverted() @ rest_center)
        center_basis_new = (target.inverted() @ waist_pose @
                            rest_waist.inverted() @ rest_center)
        location, rotation, _ = center_basis_new.decompose()
        for index in range(3):
            center_locations[index].append(location[index])
        for index in range(4):
            center_rotations[index].append(rotation[index])
        for name in ik_targets:
            bone = armature.data.bones[name]
            parent = bone.parent
            rest_ik = bone.matrix_local.copy()
            rest_parent = parent.matrix_local.copy()
            parent_basis = (
                Matrix.Translation([_evaluate(curves,
                                              f'pose.bones["{parent.name}"].location',
                                              index, frame)
                                    for index in range(3)]) @
                _quaternion_at(curves,
                               f'pose.bones["{parent.name}"].rotation_quaternion',
                               frame).to_matrix().to_4x4())
            ik_basis = Matrix.Translation([
                _evaluate(curves, f'pose.bones["{name}"].location', index, frame)
                for index in range(3)])
            chain = (rest_master @ master_basis @ rest_master.inverted() @
                     rest_parent @ parent_basis @ rest_parent.inverted() @ rest_ik)
            ik_pose = chain @ ik_basis
            chain_after = (rest_master @ master_after @ rest_master.inverted() @
                           rest_parent @ parent_basis @ rest_parent.inverted() @ rest_ik)
            new_basis = chain_after.inverted() @ ik_pose
            for index in range(3):
                ik_locations[name][index].append(new_basis.translation[index])
    if has_rotation:
        _write_curves(action, center_rotation_path, frames, center_rotations)
    _write_curves(action, center_location_path, frames, center_locations)
    for name in ik_targets:
        _write_curves(action, f'pose.bones["{name}"].location', frames,
                      ik_locations[name])
    for prop in ('rotation_quaternion', 'rotation_euler', 'rotation_axis_angle'):
        _remove_curves(action, f'pose.bones["{_WAIST_BONE}"].{prop}')
    _remove_curves(action, master_path)
    return True


def _reset_transferred_pose(armature):
    waist = armature.pose.bones.get(_WAIST_BONE)
    if waist is not None:
        if waist.rotation_mode == 'QUATERNION':
            waist.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        elif waist.rotation_mode == 'AXIS_ANGLE':
            waist.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
        else:
            waist.rotation_euler = (0.0, 0.0, 0.0)
    master = armature.pose.bones.get(_MASTER_BONE)
    if master is not None:
        master.location = (0.0, 0.0, 0.0)


class RGBMOCAP_OT_standardize(bpy.types.Operator):
    bl_idname = 'rgbmocap.standardize'
    bl_label = '标准化'
    bl_description = ('把 腰 的旋转关键帧移到 センター，把 全ての親 的位移移到 センター，'
                      '并清除原通道')
    bl_options = {'REGISTER', 'UNDO'}

    def check_props(self, context):
        obj = context.active_object
        if obj is None or obj not in context.selected_objects:
            selected = list(context.selected_objects)
            obj = selected[0] if selected else obj
        if obj is None:
            self.report({'ERROR'}, '请先选中 MMD 模型对象。')
            return None
        root, armature = mmd.find_root_and_armature(obj)
        if armature is None:
            self.report({'ERROR'}, '未找到 MMD 骨架。')
            return None
        for name in (_CENTER_BONE, _WAIST_BONE, _MASTER_BONE):
            if armature.data.bones.get(name) is None:
                self.report({'ERROR'}, f'骨架缺少 {name} 骨骼。')
                return None
        action = armature.animation_data.action if armature.animation_data else None
        if action is None:
            self.report({'ERROR'}, '骨架没有可标准化的动作。')
            return None
        return armature, action

    def execute(self, context):
        checked = self.check_props(context)
        if checked is None:
            return {'CANCELLED'}
        armature, action = checked
        _move_standardize_channels(action, armature)
        _reset_transferred_pose(armature)
        context.view_layer.update()
        self.report({'INFO'}, '已把 腰 旋转与 全ての親 位移移动到 センター。')
        return {'FINISHED'}


class RGBMOCAP_OT_fk_to_ik(bpy.types.Operator):
    bl_idname = 'rgbmocap.fk_to_ik'
    bl_label = 'FK->IK'
    bl_description = ('在临时场景里拷贝骨架、删掉上半身子孙骨骼后把腿部 FK 转为 MMD IK，'
                      '再把 IK 关键帧写回原骨架')
    bl_options = {'REGISTER', 'UNDO'}

    def check_props(self, context):
        if not mmd.mmd_tools_enabled():
            self.report({'ERROR'}, '请先启用 MMD Tools 插件。')
            return None
        obj = context.active_object
        if obj is None or obj not in context.selected_objects:
            selected = list(context.selected_objects)
            obj = selected[0] if selected else obj
        if obj is None:
            self.report({'ERROR'}, '请先选中 MMD 模型对象。')
            return None
        root, armature = mmd.find_root_and_armature(obj)
        if armature is None:
            self.report({'ERROR'}, '未找到 MMD 骨架。')
            return None
        for side in ('.L', '.R'):
            if (armature.pose.bones.get(f'足ＩＫ{side}') is None or
                    armature.pose.bones.get(f'ひざ{side}') is None or
                    armature.pose.bones.get(f'足首{side}') is None):
                self.report({'ERROR'}, '骨架缺少 MMD 腿部 IK 骨骼。')
                return None
        scene = context.scene
        return armature, int(scene.frame_start), int(scene.frame_end)

    def execute(self, context):
        checked = self.check_props(context)
        if checked is None:
            return {'CANCELLED'}
        armature, start, end = checked
        original_scene = context.scene
        source_action = (armature.animation_data.action
                         if armature.animation_data else None)

        bpy.ops.scene.new(type='NEW')
        tmp_scene = context.scene
        tmp_scene.frame_start = start
        tmp_scene.frame_end = end
        tmp = armature.copy()
        tmp.data = armature.data.copy()
        tmp.name = armature.name + '_fkik_tmp'
        tmp.parent = None
        tmp.matrix_world = armature.matrix_world.copy()
        tmp_scene.collection.objects.link(tmp)
        tmp_animation = tmp.animation_data_create()
        tmp_action = source_action.copy() if source_action else None
        if tmp_action is not None:
            tmp_animation.action = tmp_action
            try:
                if getattr(tmp_action, 'slots', None):
                    tmp_animation.action_slot = tmp_action.slots[0]
            except Exception:
                pass

        self._state = dict(
            context=context,
            original_scene=original_scene,
            armature=armature,
            tmp=tmp,
            tmp_scene=tmp_scene,
            tmp_action=tmp_action,
            start=start,
            end=end,
            frame=start,
            records={'.L': [], '.R': []},
        )
        try:
            tmp.select_set(True)
            context.view_layer.objects.active = tmp
            _remove_upper_body_children(context, tmp)
        except Exception:
            self._cleanup(return_to_original=True)
            raise

        total = end - start + 1
        set_bone_progress(True, 0.0, f'转换中 0%（0/{total} 帧）')
        if bpy.app.background or context.window is None:
            for frame in range(start, end + 1):
                _convert_frame(tmp, tmp_scene, frame, self._state['records'])
                self._state['frame'] = frame + 1
            self._finish(context)
            return {'FINISHED'}
        self._timer = context.window_manager.event_timer_add(
            0.05, window=context.window)
        context.window_manager.modal_handler_add(self)
        tag_redraw(context)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type != 'TIMER':
            return {'RUNNING_MODAL'}
        state = self._state
        total = state['end'] - state['start'] + 1
        step = max(1, total // 100)
        for _ in range(step):
            if state['frame'] > state['end']:
                break
            _convert_frame(state['tmp'], state['tmp_scene'],
                           state['frame'], state['records'])
            state['frame'] += 1
        if state['frame'] > state['end']:
            self._finish(context)
            return {'FINISHED'}
        done = state['frame'] - state['start']
        factor = done / total
        set_bone_progress(True, factor,
                          f'转换中 {int(factor * 100)}%（{done}/{total} 帧）')
        tag_redraw(context)
        return {'RUNNING_MODAL'}

    def _finish(self, context):
        state = self._state
        self._cleanup(return_to_original=True)
        _write_back_leg_ik(state['armature'], state['records'])
        _remove_leg_fcurves(state['armature'])
        _set_leg_ik(state['armature'], True)
        for side in ('.L', '.R'):
            ik = state['armature'].pose.bones.get(f'足ＩＫ{side}')
            if ik is not None:
                ik.mmd_ik_toggle = True
        set_bone_progress(False)
        tag_redraw(context)
        self.report({'INFO'},
                    f'已在临时场景完成 FK->IK（{state["start"]}..{state["end"]} 帧），'
                    '并把 IK 关键帧写回原骨架。')

    def _cleanup(self, return_to_original=True):
        state = self._state
        if getattr(self, '_timer', None) is not None:
            try:
                state['context'].window_manager.event_timer_remove(self._timer)
            except Exception:
                pass
            self._timer = None
        if return_to_original and state.get('original_scene') is not None:
            window = getattr(state['context'], 'window', None)
            if window is not None:
                window.scene = state['original_scene']
        tmp = state.get('tmp')
        if tmp is not None:
            tmp_data = tmp.data
            bpy.data.objects.remove(tmp, do_unlink=True)
            bpy.data.armatures.remove(tmp_data)
            state['tmp'] = None
        if state.get('tmp_scene') is not None:
            bpy.data.scenes.remove(state['tmp_scene'])
            state['tmp_scene'] = None
        if state.get('tmp_action') is not None:
            bpy.data.actions.remove(state['tmp_action'])
            state['tmp_action'] = None
