"""Bone-level operators: standardize channels and FK to MMD IK conversion."""
import functools
import time

import bpy
from mathutils import Matrix, Quaternion, Vector

from ..core import finger_limits, mmd

_BONE_PROGRESS = {
    'active': False,
    'factor': 0.0,
    'text': '',
}



def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()

        result = func(*args, **kwargs)

        end = time.perf_counter()

        print(f"{func.__name__} 执行耗时: {end - start:.6f} 秒")

        return result

    return wrapper

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
    """清理腿部 FK 通道。

    膝盖的旋转由 ひざ 的 IK 完全接管，删掉后导出 VMD 时 MMD 不会再先
    吃一遍 FK 旋转、导致膝盖方向与 Blender 不一致。
    大腿（足）、足首 与脚趾的关键帧保留：足/足首 的旋转是 IK 的弯曲平面
    与脚掌 roll 参考，脚趾旋转决定脚尖姿态。
    """
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        return
    rotation_paths = ('rotation_quaternion', 'rotation_euler',
                      'rotation_axis_angle')
    chain = [f'{name}{side}' for side in ('.L', '.R')
             for name in ('ひざ', '足首', 'つま先', '足先EX')]
    ik_driven = {f'ひざ{side}' for side in ('.L', '.R')}
    for fcurve in list(action.fcurves):
        path = fcurve.data_path
        owner = next((name for name in chain
                      if path.startswith(f'pose.bones["{name}"]')), None)
        if owner is None:
            continue
        if path.endswith(rotation_paths):
            if owner in ik_driven:
                action.fcurves.remove(fcurve)
            continue
        action.fcurves.remove(fcurve)


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


def _world_rotation_at(curves, armature, names, frame):
    """按静息关系与曲线求一条骨骼链末端的世界旋转。"""
    rotation = None
    previous = None
    for name in names:
        bone = armature.data.bones.get(name)
        if bone is None:
            return Quaternion()
        rest = bone.matrix_local.to_quaternion()
        if rotation is None:
            rotation = rest
        else:
            rotation = rotation @ (previous.inverted() @ rest)
        rotation = rotation @ _quaternion_at(
            curves, f'pose.bones["{name}"].rotation_quaternion', frame)
        previous = rest
    return rotation if rotation is not None else Quaternion()


def _distribute_waist_rotation(action, armature):
    """把 腰 的旋转增量到 上半身/下半身，支点补偿写到 センター。"""
    waist_path = f'pose.bones["{_WAIST_BONE}"].rotation_quaternion'
    if not _curves(action, waist_path):
        return False
    rest_waist = armature.data.bones[_WAIST_BONE].matrix_local.copy()
    rest_waist_q = rest_waist.to_quaternion()
    curves = _curve_map(action)
    waist_frames = _key_times(action, waist_path)
    for target in ('下半身', '上半身'):
        bone = armature.data.bones.get(target)
        if bone is None or armature.pose.bones.get(target) is None:
            continue
        target_path = f'pose.bones["{target}"].rotation_quaternion'
        frames = sorted(waist_frames | _key_times(action, target_path))
        rest_target_q = bone.matrix_local.to_quaternion()
        quaternions = []
        for frame in frames:
            q_waist = _quaternion_at(curves, waist_path, frame)
            q_target = _quaternion_at(curves, target_path, frame)
            delta = (rest_target_q.inverted() @ rest_waist_q @ q_waist @
                     rest_waist_q.inverted() @ rest_target_q)
            quaternions.append((delta @ q_target).normalized())
        _write_curves(action, target_path, frames,
                      [[q[i] for q in quaternions] for i in range(4)])

    # 旋转支点从 腰 头移到 下半身 头，整段身体会平移同一个世界位移；
    # 下半身/上半身 不接收位置通道，把反向位移写进 センター。
    lower = armature.data.bones.get('下半身')
    center = armature.data.bones.get(_CENTER_BONE)
    master = armature.data.bones.get(_MASTER_BONE)
    if lower is not None and center is not None and master is not None:
        waist_bone = armature.data.bones[_WAIST_BONE]
        ancestors = []
        parent = waist_bone.parent
        while parent is not None:
            ancestors.append(parent.name)
            parent = parent.parent
        ancestors.reverse()
        rel_lower = waist_bone.matrix_local.inverted() @ lower.matrix_local
        lower_offset = rel_lower.translation.copy()
        rest_master = master.matrix_local.to_quaternion()
        rest_center = center.matrix_local.to_quaternion()
        center_path = f'pose.bones["{_CENTER_BONE}"].location'
        center_frames = sorted(waist_frames | _key_times(action, center_path))
        center_locations = ([], [], [])
        for frame in center_frames:
            q_waist = _quaternion_at(curves, waist_path, frame)
            if ancestors:
                base = _world_rotation_at(curves, armature, ancestors, frame)
                base = base @ (armature.data.bones[ancestors[-1]].matrix_local
                               .to_quaternion().inverted() @ rest_waist_q)
            else:
                base = rest_waist_q.copy()
            shift = base.to_matrix() @ (
                (q_waist.to_matrix() - Matrix.Identity(3)) @ lower_offset)
            master_rotation = _world_rotation_at(
                curves, armature, (_MASTER_BONE,), frame)
            mapping = master_rotation @ (rest_master.inverted() @ rest_center)
            correction = mapping.inverted() @ shift
            for index in range(3):
                center_locations[index].append(
                    _evaluate(curves, center_path, index, frame) +
                    correction[index])
        _write_curves(action, center_path, center_frames, center_locations)

    for prop in ('rotation_quaternion', 'rotation_euler', 'rotation_axis_angle'):
        _remove_curves(action, f'pose.bones["{_WAIST_BONE}"].{prop}')
    return True


def _move_standardize_channels(action, armature):
    """把 腰 旋转增量到 上半身/下半身，把 全ての親 位移搬到 センター。"""
    moved = _distribute_waist_rotation(action, armature)
    master_path = f'pose.bones["{_MASTER_BONE}"].location'
    master_rotation_path = f'pose.bones["{_MASTER_BONE}"].rotation_quaternion'
    center_location_path = f'pose.bones["{_CENTER_BONE}"].location'
    center_rotation_path = f'pose.bones["{_CENTER_BONE}"].rotation_quaternion'
    if not _curves(action, master_path):
        return moved
    ik_targets = []
    for side in ('.L', '.R'):
        name = f'足ＩＫ{side}'
        bone = armature.data.bones.get(name)
        if bone is None or bone.parent is None:
            continue
        if _curves(action, f'pose.bones["{name}"].location'):
            ik_targets.append(name)
    frames = set()
    for path in (master_path, master_rotation_path,
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
    center_locations = ([], [], [])
    ik_locations = {name: ([], [], []) for name in ik_targets}
    for frame in frames:
        master_location = Vector([_evaluate(curves, master_path, index, frame)
                                  for index in range(3)])
        master_rotation = _quaternion_at(curves, master_rotation_path, frame)
        center_basis = (
            Matrix.Translation([_evaluate(curves, center_location_path, index, frame)
                                for index in range(3)]) @
            _quaternion_at(curves, center_rotation_path, frame).to_matrix().to_4x4())
        master_basis = (Matrix.Translation(master_location) @
                        master_rotation.to_matrix().to_4x4())
        master_after = master_rotation.to_matrix().to_4x4()
        center_pose = (rest_master @ master_basis @ rest_master.inverted() @
                       rest_center @ center_basis)
        target = (rest_master @ master_after @ rest_master.inverted() @ rest_center)
        location = (target.inverted() @ center_pose).translation
        for index in range(3):
            center_locations[index].append(location[index])
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
    _write_curves(action, center_location_path, frames, center_locations)
    for name in ik_targets:
        _write_curves(action, f'pose.bones["{name}"].location', frames,
                      ik_locations[name])
    _remove_curves(action, master_path)
    return True


def _convert_bone_rotation(action, armature, source_name, target_name):
    """把 source_name 的旋转曲线按世界等价转到 target_name。

    两条骨骼只差一个静息朝向的共轭变换：
    q_target = R_target⁻¹ · R_source · q_source · R_source⁻¹ · R_target
    纯曲线求值，不需要逐帧更新场景。
    """
    source_path = f'pose.bones["{source_name}"].rotation_quaternion'
    if not _curves(action, source_path):
        return False
    source_bone = armature.data.bones.get(source_name)
    target_bone = armature.data.bones.get(target_name)
    if source_bone is None or target_bone is None:
        return False
    world_rotation = armature.matrix_world.to_quaternion()
    source_rest = world_rotation @ source_bone.matrix_local.to_quaternion()
    target_rest = world_rotation @ target_bone.matrix_local.to_quaternion()
    left = target_rest.inverted() @ source_rest
    right = source_rest.inverted() @ target_rest
    curves = _curve_map(action)
    frames = sorted(_key_times(action, source_path))
    quaternions = []
    for frame in frames:
        q_source = _quaternion_at(curves, source_path, frame)
        quaternions.append((left @ q_source @ right).normalized())
    if not quaternions:
        return False
    _write_curves(action, f'pose.bones["{target_name}"].rotation_quaternion',
                  frames, [[q[i] for q in quaternions] for i in range(4)])
    for prop in ('rotation_quaternion', 'rotation_euler',
                 'rotation_axis_angle'):
        _remove_curves(action, f'pose.bones["{source_name}"].{prop}')
    source_pose = armature.pose.bones.get(source_name)
    if source_pose is not None:
        source_pose.rotation_mode = 'QUATERNION'
        source_pose.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        source_pose.rotation_euler = (0.0, 0.0, 0.0)
        source_pose.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
    return True


_TOE_BONES = ('つま先', '足先EX')

@timeit
def convert_toe_rotation(armature, enabled):
    """在 つま先 与 足先EX 之间搬运旋转；返回 (成功, 错误信息)。

    开启：つま先 → 足先EX；关闭：足先EX → つま先。
    两侧骨骼必须齐全，缺一侧按两侧都缺处理；没有动作或没有可转换的
    关键帧时静默跳过（这个开关只表示意图，不保证一定有数据）。
    """
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        return True, ''
    bones = armature.data.bones
    missing = [f'{stem}{side}' for side in ('.L', '.R') for stem in _TOE_BONES
               if bones.get(f'{stem}{side}') is None]
    if missing:
        return False, '缺少骨骼：' + '、'.join(missing) + '（缺一即两侧都缺）'
    source, target = ((_TOE_BONES[0], _TOE_BONES[1]) if enabled
                      else (_TOE_BONES[1], _TOE_BONES[0]))
    for side in ('.L', '.R'):
        _convert_bone_rotation(action, armature, source + side, target + side)
    return True, ''

@timeit
def standardize_channels(armature):
    """把 腰 旋转增量到 上半身/下半身，位移搬到 センター，并开启 腰キャンセル。"""
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        return False
    _move_standardize_channels(action, armature)
    _reset_transferred_pose(armature)
    _compensate_waist_cancel(action, armature)
    return True

@timeit
def _compensate_waist_cancel(action, armature):
    """恢复 腰キャンセル 的追加変形，并把被取消的 腰 旋转补偿到 足。

    MMD 的 腰キャンセル 会取消 腰 的局部旋转，但约束状态不会写进 VMD。
    这里把被取消的旋转按静息关系共轭到 足，写回它的局部旋转，使导出后
    开启追加変形时腿的朝向仍与烘焙结果一致。
    """
    pairs = []
    for side in ('.L', '.R'):
        cancel = armature.pose.bones.get('腰キャンセル' + side)
        foot = armature.pose.bones.get('足' + side)
        if cancel is None or foot is None or foot.parent is None:
            continue
        constraint = cancel.constraints.get('mmd_additional_rotation')
        if constraint is None or constraint.type != 'TRANSFORM':
            continue
        influence = getattr(getattr(cancel, 'mmd_bone', None),
                            'additional_transform_influence', -1.0)
        if abs(influence + 1.0) > 1e-6:
            # 只处理完整取反的追加変形，其余保持静音。
            continue
        pairs.append((constraint, foot))
    if not pairs:
        return False
    if action.get('flowhmr2vmd_waist_cancel'):
        # 这个动作已经补偿过，避免重复叠加。
        for constraint, _ in pairs:
            constraint.mute = False
        return True
    waist_path = f'pose.bones["{_WAIST_BONE}"].rotation_quaternion'
    if not _curves(action, waist_path):
        for constraint, _ in pairs:
            constraint.mute = False
        return True
    frames = sorted({int(round(time)) for time in set().union(*(
        _key_times(action, f'pose.bones["{foot.name}"].rotation_quaternion')
        for _, foot in pairs))})
    if not frames:
        return False
    curves = _curve_map(action)
    rest_waist = armature.data.bones[_WAIST_BONE].matrix_local.to_quaternion()
    for constraint, foot in pairs:
        constraint.mute = False
        foot_path = f'pose.bones["{foot.name}"].rotation_quaternion'
        rest_foot = foot.bone.matrix_local.to_quaternion()
        rest_leg = rest_waist.inverted() @ rest_foot
        quaternions = []
        for frame in frames:
            q_waist = _quaternion_at(curves, waist_path, frame)
            q_foot = _quaternion_at(curves, foot_path, frame)
            quaternions.append(
                (rest_leg.inverted() @ q_waist @ rest_leg @ q_foot).normalized())
        _write_curves(action, foot_path, frames,
                      [[q[i] for q in quaternions] for i in range(4)],
                      group='waist_cancel')
    action['flowhmr2vmd_waist_cancel'] = True
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


class FLOWHMR2VMD_OT_standardize(bpy.types.Operator):
    bl_idname = 'flowhmr2vmd.standardize'
    bl_label = '标准化'
    bl_description = ('把 腰 旋转增量到 上半身/下半身，把 全ての親 位移移到 センター，'
                      '并开启 腰キャンセル')
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
        standardize_channels(armature)
        context.view_layer.update()
        self.report({'INFO'}, '已把 腰 旋转增量到 上半身/下半身，位移移到 センター，并开启 腰キャンセル。')
        return {'FINISHED'}


class FLOWHMR2VMD_OT_fk_to_ik(bpy.types.Operator):
    bl_idname = 'flowhmr2vmd.fk_to_ik'
    bl_label = 'FK->IK'
    bl_description = '将腿部驱动方式由FK转换为IK，以便后续调整动作。'
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
        self.report({'INFO'}, f'已完成 FK->IK')

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


_FINGER_BASE_NAMES = (
    '親指０', '親指１', '親指２',
    '人指１', '人指２', '人指３',
    '中指１', '中指２', '中指３',
    '薬指１', '薬指２', '薬指３',
    '小指１', '小指２', '小指３',
)


def _select_pose_bone(pose_bone, selected):
    """跨版本选择骨骼：3.6 用 Bone.select，5.0 起改为 PoseBone.select。"""
    if hasattr(pose_bone, 'select'):
        pose_bone.select = selected
    else:
        pose_bone.bone.select = selected


class FLOWHMR2VMD_OT_bake_fingers(bpy.types.Operator):
    bl_idname = 'flowhmr2vmd.bake_fingers'
    bl_label = '烘焙手指动作'
    bl_description = '将手指骨骼约束与驱动器的计算结果烘焙到当前动作中'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if not mmd.mmd_tools_enabled():
            self.report({'ERROR'}, '请先启用 MMD Tools 插件。')
            return {'CANCELLED'}
        selected = list(context.selected_objects)
        obj = context.active_object if context.active_object in selected else (
            selected[0] if selected else None)
        if obj is None:
            self.report({'ERROR'}, '请选择 MMD 模型')
            return {'CANCELLED'}
        root, armature = mmd.find_root_and_armature(obj)
        if armature is None:
            self.report({'ERROR'}, '请选择 MMD 模型')
            return {'CANCELLED'}
        state = mmd.read_state(root)
        if (not state[mmd.STATE_FINGER_LIMITS] and
                not state[mmd.STATE_FINGER_DRIVER]):
            self.report({'ERROR'}, '请开启手指约束')
            return {'CANCELLED'}
        fingers = [armature.pose.bones[f'{base}{side}']
                   for side in ('.L', '.R') for base in _FINGER_BASE_NAMES
                   if armature.pose.bones.get(f'{base}{side}') is not None]
        if not fingers:
            self.report({'ERROR'}, '骨架缺少手指骨骼。')
            return {'CANCELLED'}

        original_scene = context.scene
        frame_end = int(original_scene.frame_end)

        bpy.ops.scene.new(type='NEW')
        tmp_scene = context.scene
        tmp_scene.frame_start = 0
        tmp_scene.frame_end = frame_end
        # 直接复用原骨架，不拷贝对象；烘焙结果写入同一个 action。
        tmp_scene.collection.objects.link(armature)
        for item in context.view_layer.objects:
            item.select_set(False)
        armature.select_set(True)
        context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='POSE')
        for pose_bone in armature.pose.bones:
            _select_pose_bone(pose_bone, False)
        for pose_bone in fingers:
            _select_pose_bone(pose_bone, True)
        bpy.ops.nla.bake(
            frame_start=0, frame_end=frame_end, visual_keying=True,
            clear_constraints=True, clear_parents=True,
            use_current_action=True, clean_curves=True,
            bake_types={'POSE'})
        bpy.ops.object.mode_set(mode='OBJECT')

        window = getattr(context, 'window', None)
        if window is not None:
            window.scene = original_scene
        bpy.data.scenes.remove(tmp_scene)
        # 烘焙后约束已被清除，根对象上的开关同步关闭。
        mmd.write_state(root, mmd.STATE_FINGER_LIMITS, False)
        mmd.write_state(root, mmd.STATE_FINGER_DRIVER, False)
        self.report({'INFO'},
                    f'已烘焙 {len(fingers)} 根手指骨骼（0..{frame_end} 帧）。')
        return {'FINISHED'}


_MODE_ITEMS = (
    ('ENABLED', '开启', '执行开启操作'),
    ('DISABLED', '关闭', '执行关闭操作'),
)


def _selected_model(context):
    selected = list(context.selected_objects)
    obj = context.active_object if context.active_object in selected else (
        selected[0] if selected else None)
    if obj is None:
        return None, None
    root, armature = mmd.find_root_and_armature(obj)
    if root is None or armature is None:
        return None, None
    return root, armature


class FLOWHMR2VMD_OT_rotate_toe(bpy.types.Operator):
    bl_idname = 'flowhmr2vmd.rotate_toe'
    bl_label = '脚尖旋转'
    bl_description = '开启后可使脚尖旋转，请根据实际效果来决定是否开启'
    bl_options = {'UNDO'}

    mode: bpy.props.EnumProperty(name='操作', items=_MODE_ITEMS,
                                 default='ENABLED')

    def execute(self, context):
        if not mmd.mmd_tools_enabled():
            self.report({'ERROR'}, '请先启用 MMD Tools 插件')
            return {'CANCELLED'}
        root, armature = _selected_model(context)
        if armature is None:
            self.report({'ERROR'}, '请选择 MMD 模型')
            return {'CANCELLED'}
        enabled = self.mode == 'ENABLED'
        ok, message = convert_toe_rotation(armature, enabled)
        if not ok:
            self.report({'ERROR'}, message)
            return {'CANCELLED'}
        mmd.write_state(root, mmd.STATE_ROTATE_TOE, enabled)
        return {'FINISHED'}


class FLOWHMR2VMD_OT_finger_limits(bpy.types.Operator):
    bl_idname = 'flowhmr2vmd.finger_limits'
    bl_label = '手指约束'
    bl_description = '开启时给手指骨骼添加限制旋转约束，关闭时删除本插件添加的约束'
    bl_options = {'UNDO'}

    mode: bpy.props.EnumProperty(name='操作', items=_MODE_ITEMS,
                                 default='ENABLED')

    def execute(self, context):
        if not mmd.mmd_tools_enabled():
            self.report({'ERROR'}, '请先启用 MMD Tools 插件')
            return {'CANCELLED'}
        root, armature = _selected_model(context)
        if armature is None:
            self.report({'ERROR'}, '请选择 MMD 模型')
            return {'CANCELLED'}
        props = getattr(context.scene, 'flowhmr2vmd_properties', None)
        if props is None:
            return {'CANCELLED'}
        if self.mode == 'ENABLED':
            missing = [f'{base}{side}'
                       for base in finger_limits.all_limits(props)
                       for side in ('.L', '.R')
                       if armature.pose.bones.get(base + side) is None]
            if missing:
                self.report({'ERROR'}, '骨架缺少手指骨骼：' + '、'.join(missing))
                return {'CANCELLED'}
            finger_limits.apply_finger_limits(armature, props)
            mmd.write_state(root, mmd.STATE_FINGER_LIMITS, True)
        else:
            finger_limits.remove_finger_drivers(armature)
            finger_limits.remove_finger_limits(armature)
            mmd.write_state(root, mmd.STATE_FINGER_DRIVER, False)
            mmd.write_state(root, mmd.STATE_FINGER_LIMITS, False)
        return {'FINISHED'}


class FLOWHMR2VMD_OT_finger_driver(bpy.types.Operator):
    bl_idname = 'flowhmr2vmd.finger_driver'
    bl_label = '手指约束驱动器'
    bl_description = '开启时给手指约束添加驱动器，关闭时移除驱动器并恢复面板范围'
    bl_options = {'UNDO'}

    mode: bpy.props.EnumProperty(name='操作', items=_MODE_ITEMS,
                                 default='ENABLED')

    def execute(self, context):
        if not mmd.mmd_tools_enabled():
            self.report({'ERROR'}, '请先启用 MMD Tools 插件')
            return {'CANCELLED'}
        root, armature = _selected_model(context)
        if armature is None:
            self.report({'ERROR'}, '请选择 MMD 模型')
            return {'CANCELLED'}
        props = getattr(context.scene, 'flowhmr2vmd_properties', None)
        if props is None:
            return {'CANCELLED'}
        state = mmd.read_state(root)
        if self.mode == 'ENABLED':
            if not state[mmd.STATE_FINGER_LIMITS]:
                self.report({'ERROR'}, '请先开启手指约束')
                return {'CANCELLED'}
            finger_limits.apply_finger_limits(armature, props)
            finger_limits.apply_finger_drivers(armature, props, context.scene)
            mmd.write_state(root, mmd.STATE_FINGER_DRIVER, True)
        else:
            finger_limits.remove_finger_drivers(armature)
            if state[mmd.STATE_FINGER_LIMITS]:
                finger_limits.apply_finger_limits(armature, props)
            mmd.write_state(root, mmd.STATE_FINGER_DRIVER, False)
        return {'FINISHED'}

