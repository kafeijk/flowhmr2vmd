"""Bone-level operators: standardize channels and FK to MMD IK conversion."""
import functools
import math
import time

import bpy
import numpy as np
from mathutils import Euler, Matrix, Quaternion, Vector

from ..core import mmd


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()

        result = func(*args, **kwargs)

        end = time.perf_counter()

        print(f"{func.__name__} 执行耗时: {end - start:.6f} 秒")

        return result

    return wrapper


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
            (f'pose.bones["つま先ＩＫ{side}"].location', 3, lambda row: row[3]),
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


def _enable_leg_ik_toggle(armature, frame):
    """清空 足ＩＫ / つま先ＩＫ 的 IK 开关通道，并在第一帧写入开启。"""
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        return
    for side in ('.L', '.R'):
        for base in ('足ＩＫ', 'つま先ＩＫ'):
            pose_bone = armature.pose.bones.get(base + side)
            if pose_bone is None or not hasattr(pose_bone, 'mmd_ik_toggle'):
                continue
            path = pose_bone.path_from_id('mmd_ik_toggle')
            for fcurve in [fc for fc in action.fcurves if fc.data_path == path]:
                action.fcurves.remove(fcurve)
            pose_bone.mmd_ik_toggle = True
            pose_bone.keyframe_insert('mmd_ik_toggle', frame=frame)


def _bake_leg_fk_from_ik(armature, start, end):
    """把 IK 解算后的腿部姿态写回 FK 旋转曲线。

    这样关闭 IK 后，腿部 FK 与开启 IK 时的姿态一致；开启 IK 时仍由
    足ＩＫ / つま先ＩＫ 驱动，不受影响。
    """
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        return
    scene = bpy.context.scene
    names = [f'{name}{side}' for side in ('.L', '.R')
             for name in ('足', 'ひざ', '足首', 'つま先', '足先EX')]
    names = [name for name in names
             if armature.pose.bones.get(name) is not None]
    frames = list(range(int(start), int(end) + 1))
    values = {name: [] for name in names}
    for frame in frames:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        for name in names:
            pose_bone = armature.pose.bones[name]
            parent = pose_bone.parent
            if parent is None:
                basis = pose_bone.bone.matrix_local.inverted() @ pose_bone.matrix
            else:
                rest = (parent.bone.matrix_local.inverted() @
                        pose_bone.bone.matrix_local)
                basis = (parent.matrix @ rest).inverted() @ pose_bone.matrix
            values[name].append(basis.to_quaternion().normalized())
    for name, quaternions in values.items():
        if not quaternions:
            continue
        _write_curves(action, f'pose.bones["{name}"].rotation_quaternion',
                      frames,
                      [[q[i] for q in quaternions] for i in range(4)],
                      group='fk_ik')
    scene.frame_set(int(start))


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


def _sample_column(curves, path, frames, index=0, default=0.0):
    """批量取单条曲线在给定帧上的值：关键帧查表，缺失的帧回退求值。"""
    fcurve = curves.get((path, index))
    if fcurve is None:
        return [default] * len(frames)
    points = fcurve.keyframe_points
    co = np.empty(len(points) * 2, np.float64)
    points.foreach_get('co', co)
    table = dict(zip(co[0::2].tolist(), co[1::2].tolist()))
    return [table[frame] if frame in table else fcurve.evaluate(frame)
            for frame in frames]


def _sample_columns(curves, path, frames):
    """批量取 path 四个通道在给定帧上的值，帧缺失时回退求值。"""
    return [_sample_column(curves, path, frames, index,
                           1.0 if index == 0 else 0.0)
            for index in range(4)]


def _remove_curves(action, path):
    for fcurve in _curves(action, path):
        action.fcurves.remove(fcurve)


def _key_styles(fcurve):
    """读取源曲线的插值信息，须在删除/新增曲线之前调用。"""
    if fcurve is None:
        return None
    points = fcurve.keyframe_points
    if not len(points):
        return None
    if all(point.interpolation == 'BEZIER' and
           point.handle_left_type == 'FREE' and
           point.handle_right_type == 'FREE'
           for point in points):
        return 'straight'
    return {round(point.co.x): point.interpolation for point in points}


def _apply_key_styles(fcurve, styles):
    """沿用源曲线的插值方式：直线贝塞尔原样重建，其余按帧复制插值类型。"""
    if styles == 'straight':
        _set_straight_bezier(fcurve)
        return
    points = fcurve.keyframe_points
    count = len(points)
    co = np.empty(count * 2, np.float64)
    points.foreach_get('co', co)
    order = {'CONSTANT': 0, 'LINEAR': 1, 'BEZIER': 2}
    values = [order.get(styles.get(round(frame), 'LINEAR'), 1)
              for frame in co[0::2]]
    points.foreach_set('interpolation', values)


def _write_curves(action, path, frames, columns, group='standardize', styles=None):
    _remove_curves(action, path)
    count = len(frames)
    co = np.empty((count, 2), np.float64)
    # 将frames中的所有帧数，批量赋值给二维数组co的第0列
    co[:, 0] = frames
    linear = [1] * count
    for index, values in enumerate(columns):
        fcurve = action.fcurves.new(path, index=index, action_group=group)
        fcurve.keyframe_points.add(count)
        co[:, 1] = values
        fcurve.keyframe_points.foreach_set('co', co.ravel())
        if styles is None:
            fcurve.keyframe_points.foreach_set('interpolation', linear)
        else:
            _apply_key_styles(fcurve, styles)
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


def _move_master_to_center(action, armature):
    """把 全ての親 的位移搬到 センター，并补偿 足ＩＫ 的位置。"""
    master_path = f'pose.bones["{_MASTER_BONE}"].location'
    master_rotation_path = f'pose.bones["{_MASTER_BONE}"].rotation_quaternion'
    center_location_path = f'pose.bones["{_CENTER_BONE}"].location'
    center_rotation_path = f'pose.bones["{_CENTER_BONE}"].rotation_quaternion'
    if not _curves(action, master_path):
        return False
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
    """
    将源骨骼相对于静息姿态的旋转差，转换到目标骨骼的局部坐标系中。

    计算源骨骼当前姿态与静息姿态在世界坐标系下的旋转差，
    再将该旋转差转换为目标骨骼局部坐标系下的旋转。

    q_target = R_target⁻¹ · R_source · q_source · R_source⁻¹ · R_target

    其中：
        R_source：源骨骼的世界静息朝向。
        R_target：目标骨骼的世界静息朝向。
        q_source：源骨骼当前相对于静息姿态的旋转。
        q_target：转换后的目标骨骼局部旋转。

    通过动画曲线求值和写入完成转换，避免约束逐帧更新场景。
    """
    source_path = f'pose.bones["{source_name}"].rotation_quaternion'
    if not _curves(action, source_path):
        return False

    source_bone = armature.data.bones.get(source_name)
    target_bone = armature.data.bones.get(target_name)
    if source_bone is None or target_bone is None:
        return False

    # 获取骨架对象的世界旋转，将两根骨骼的静息朝向转换到世界坐标系。
    world_rotation = armature.matrix_world.to_quaternion()
    source_rest = world_rotation @ source_bone.matrix_local.to_quaternion()
    target_rest = world_rotation @ target_bone.matrix_local.to_quaternion()

    # 构建源骨骼与目标骨骼之间的静息朝向转换关系。
    # left：目标骨骼世界静息朝向的逆旋转 × 源骨骼世界静息朝向。
    # right：源骨骼世界静息朝向的逆旋转 × 目标骨骼世界静息朝向。
    left = target_rest.inverted() @ source_rest
    right = source_rest.inverted() @ target_rest

    # 获取源骨骼动画曲线及所有关键帧时间。
    curves = _curve_map(action)
    frames = sorted(_key_times(action, source_path))
    # 批量读取源骨骼四元数通道，并转换到目标骨骼的局部坐标系。
    columns = _sample_columns(curves, source_path, frames)
    quaternions = [(left @ Quaternion([column[i] for column in columns])
                    .normalized() @ right).normalized()
                   for i in range(len(frames))]

    if not quaternions:
        return False

    # 将转换后的四元数写入目标骨骼动画曲线。
    styles = _key_styles(curves.get((source_path, 0)))
    _write_curves(action, f'pose.bones["{target_name}"].rotation_quaternion',
                  frames, [[q[i] for q in quaternions] for i in range(4)],
                  styles=styles)

    # 删除源骨骼的所有旋转动画曲线，避免重复驱动。
    for prop in ('rotation_quaternion', 'rotation_euler',
                 'rotation_axis_angle'):
        _remove_curves(action, f'pose.bones["{source_name}"].{prop}')

    # 将源骨骼当前姿态恢复为静置姿态。
    source_pose = armature.pose.bones.get(source_name)
    if source_pose is not None:
        source_pose.rotation_mode = 'QUATERNION'
        source_pose.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        source_pose.rotation_euler = (0.0, 0.0, 0.0)
        source_pose.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)

    return True


_TOE_BONE = 'つま先'
_EX_BONE = '足先EX'
_EX_CUSTOM_PROPS = ('w', 'x', 'y', 'z')


def _ex_custom_path(side, component):
    return f'pose.bones["{_EX_BONE}{side}"]["flowhmr_ex_q{component}"]'


def _reset_bone_rotation(armature, bone_name):
    """恢复骨骼静置姿态。"""
    pose_bone = armature.pose.bones.get(bone_name)
    if pose_bone is None:
        return
    if pose_bone.rotation_mode == 'QUATERNION':
        pose_bone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
    elif pose_bone.rotation_mode == 'AXIS_ANGLE':
        pose_bone.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
    else:
        pose_bone.rotation_euler = (0.0, 0.0, 0.0)


def clear_ex_bone_curves(armature, side=None):
    """清空 足先EX 的骨骼关键帧并把骨骼姿态归位，保留自定义属性通道。"""
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        return False
    sides = (side,) if side else ('.L', '.R')
    cleared = False
    for value in sides:
        prefix = f'pose.bones["{_EX_BONE}{value}"]'
        for fcurve in [fc for fc in action.fcurves
                       if fc.data_path.startswith(prefix + '.')]:
            action.fcurves.remove(fcurve)
            cleared = True
        # 删除曲线不会改变当前姿态值，必须显式归位。
        _reset_bone_rotation(armature, _EX_BONE + value)
    return cleared


def _write_custom_property_curve(pose_bone, prop_name, frames, values):
    """写入骨骼自定义属性曲线；用 keyframe_insert 建立正确的 action slot。"""
    action = pose_bone.id_data.animation_data.action
    data_path = f'pose.bones["{pose_bone.name}"]["{prop_name}"]'
    for fcurve in [fc for fc in action.fcurves if fc.data_path == data_path]:
        action.fcurves.remove(fcurve)
    pose_bone[prop_name] = values[0]
    pose_bone.keyframe_insert(f'["{prop_name}"]', frame=frames[0])
    fcurve = next((fc for fc in action.fcurves
                   if fc.data_path == data_path and fc.array_index == 0), None)
    if fcurve is None:
        return False
    fcurve.keyframe_points.clear()
    fcurve.keyframe_points.add(len(frames))
    for point, frame, value in zip(fcurve.keyframe_points, frames, values):
        point.co = (frame, value)
        point.interpolation = 'LINEAR'
    fcurve.update()
    return True


def store_ex_to_ex_custom(armature):
    """把 足先EX 的旋转曲线备份到自定义属性，并清空骨骼关键帧。"""
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        return False
    stored = False
    for side in ('.L', '.R'):
        if armature.pose.bones.get(_EX_BONE + side) is None:
            continue
        pose_bone = armature.pose.bones[_EX_BONE + side]
        rotation_path = f'pose.bones["{_EX_BONE}{side}"].rotation_quaternion'
        if not _curves(action, rotation_path):
            continue
        frames = sorted(_key_times(action, rotation_path))
        if not frames:
            continue
        curves = _curve_map(action)
        for index, component in enumerate(_EX_CUSTOM_PROPS):
            prop_name = f'flowhmr_ex_q{component}'
            if pose_bone.get(prop_name) is None:
                pose_bone[prop_name] = 1.0 if index == 0 else 0.0
            values = [_evaluate(curves, rotation_path, index, frame,
                                1.0 if index == 0 else 0.0)
                      for frame in frames]
            _write_custom_property_curve(pose_bone, prop_name, frames, values)
        clear_ex_bone_curves(armature, side)
        stored = True
    return stored


def restore_ex_from_ex_custom(armature):
    """把自定义属性里的脚尖旋转恢复到 足先EX 骨骼曲线。"""
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        return False
    restored = False
    for side in ('.L', '.R'):
        if armature.pose.bones.get(_EX_BONE + side) is None:
            continue
        paths = [_ex_custom_path(side, component)
                 for component in _EX_CUSTOM_PROPS]
        if not all(_curves(action, path) for path in paths):
            continue
        frames = sorted(set().union(*(_key_times(action, path)
                                      for path in paths)))
        if not frames:
            continue
        curves = _curve_map(action)
        columns = [
            _sample_column(curves, path, frames,
                           default=1.0 if component == 'w' else 0.0)
            for component, path in zip(_EX_CUSTOM_PROPS, paths)
        ]
        styles = _key_styles(curves.get((paths[0], 0)))
        _write_curves(action,
                      f'pose.bones["{_EX_BONE}{side}"].rotation_quaternion',
                      frames, columns, group='toe',
                      styles=styles)
        restored = True
    return restored


@timeit
def convert_toe_rotation(armature, enabled):
    """
    在 つま先 与 足先EX 之间转移旋转数据；当不存在 つま先 时，使用 足先EX 的自定义属性作为替代。

    开启：
    - つま先 → 足先EX，清空 つま先 的动画曲线并归位。
    - 不存在 つま先 时：自定义属性 → 足先EX。

    关闭：
    - 足先EX → つま先，清空 足先EX 的动画曲线并归位。
    - 不存在 つま先 时：足先EX → 自定义属性，清空 足先EX 的动画曲线并归位。
    """
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        return True, ''

    bones = armature.data.bones
    has_toe = all(bones.get(f'{_TOE_BONE}{side}') is not None for side in ('.L', '.R'))
    has_ex = all(bones.get(f'{_EX_BONE}{side}') is not None for side in ('.L', '.R'))
    # 缺少 足先EX 时开启脚尖旋转，提示错误
    if not has_ex and enabled:
        return False, '骨架缺少「足先EX」骨骼，无法开启脚尖旋转。'
    # 缺少 足先EX 时关闭脚尖旋转，无影响
    if not has_ex:
        return True, ''

    if has_toe:
        source, target = (_TOE_BONE, _EX_BONE) if enabled else (_EX_BONE, _TOE_BONE)
        for side in ('.L', '.R'):
            # 有 つま先 就正常与 足先EX 交换关键帧，没有 足先EX 时跳过即可
            _convert_bone_rotation(action, armature, source + side, target + side)
            # 开启时归位 つま先，关闭时归位 足先EX（对应语义均为源骨骼）
            _reset_bone_rotation(armature, source + side)
        return True, ''
    if enabled:
        # 没有 つま先 时，足先EX 从自定义属性中获取关键帧
        restore_ex_from_ex_custom(armature)
    else:
        # 关闭时把 足先EX 的关键帧存入自定义属性，再清空骨骼动画并归位
        store_ex_to_ex_custom(armature)
        clear_ex_bone_curves(armature)
    return True, ''


def fk_to_ik(armature, start, end):
    """同步执行 FK->IK：转换腿部、写入 IK 关键帧并注册 toggle。"""
    if not mmd.mmd_tools_enabled():
        return False
    for side in ('.L', '.R'):
        if (armature.pose.bones.get(f'足ＩＫ{side}') is None or
                armature.pose.bones.get(f'ひざ{side}') is None or
                armature.pose.bones.get(f'足首{side}') is None):
            return False
    source_action = (armature.animation_data.action
                     if armature.animation_data else None)
    if source_action is None:
        return False
    context = bpy.context
    original_scene = context.scene
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
    tmp_action = source_action.copy()
    tmp_animation.action = tmp_action
    try:
        if getattr(tmp_action, 'slots', None):
            tmp_animation.action_slot = tmp_action.slots[0]
    except Exception:
        pass
    records = {'.L': [], '.R': []}
    tmp_data = tmp.data
    try:
        tmp.select_set(True)
        context.view_layer.objects.active = tmp
        _remove_upper_body_children(context, tmp)
        for frame in range(int(start), int(end) + 1):
            _convert_frame(tmp, tmp_scene, frame, records)
    finally:
        window = getattr(context, 'window', None)
        if window is not None:
            window.scene = original_scene
        bpy.data.objects.remove(tmp, do_unlink=True)
        bpy.data.armatures.remove(tmp_data)
        bpy.data.scenes.remove(tmp_scene)
        bpy.data.actions.remove(tmp_action)
    _write_back_leg_ik(armature, records)
    _remove_leg_fcurves(armature)
    _enable_leg_ik_toggle(armature, start)
    _set_leg_ik(armature, True)
    return True


def _remove_knee_fcurves(armature):
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        return False
    removed = False
    for side in ('.L', '.R'):
        prefix = f'pose.bones["ひざ{side}"]'
        for fcurve in [fc for fc in action.fcurves
                       if fc.data_path.startswith(prefix)]:
            action.fcurves.remove(fcurve)
            removed = True
    return removed


@timeit
def standardize_channels(armature):
    """按步骤标准化：全亲→センター、全亲复位、腰→上下半身、腰复位、腰キャンセル、FK→IK。"""
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        return False
    _move_master_to_center(action, armature)
    _reset_master_pose(armature)
    _distribute_waist_rotation(action, armature)
    _reset_waist_pose(armature)
    _compensate_waist_cancel(action, armature)
    scene = bpy.context.scene
    start = int(scene.frame_start)
    end = int(scene.frame_end)
    if fk_to_ik(armature, start, end):
        _bake_leg_fk_from_ik(armature, start, end)
        _remove_knee_fcurves(armature)
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


def _reset_master_pose(armature):
    master = armature.pose.bones.get(_MASTER_BONE)
    if master is not None:
        master.location = (0.0, 0.0, 0.0)


def _reset_waist_pose(armature):
    waist = armature.pose.bones.get(_WAIST_BONE)
    if waist is None:
        return
    if waist.rotation_mode == 'QUATERNION':
        waist.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
    elif waist.rotation_mode == 'AXIS_ANGLE':
        waist.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
    else:
        waist.rotation_euler = (0.0, 0.0, 0.0)


# (骨骼链, 保留的局部旋转轴, 接受补偿的骨骼序号；-1 表示不补偿)
_FINGER_CHAINS = (
    ('親指０', '親指１', '親指２', 'Z', 1),
    ('人指１', '人指２', '人指３', 'X', 0),
    ('中指１', '中指２', '中指３', 'X', 0),
    ('薬指１', '薬指２', '薬指３', 'X', 0),
    ('小指１', '小指２', '小指３', 'X', 0),
)


def _keep_axis_rotation(quaternion, axis):
    """只保留局部欧拉 XYZ 分解中的指定轴旋转。"""
    euler = quaternion.to_euler('XYZ')
    values = {'X': euler.x, 'Y': euler.y, 'Z': euler.z}
    return Euler(tuple(values[name] if name == axis else 0.0
                       for name in 'XYZ'), 'XYZ').to_quaternion()


_COMPENSATION_LIMIT_DEG = 80.0


def _compensation_weight(bend_degrees):
    """弯曲越大补偿越小；达到 80° 及以上完全不补偿。"""
    return min(max(1.0 - bend_degrees / _COMPENSATION_LIMIT_DEG, 0.0), 1.0)


def _rest_relative_quaternion(bone):
    if bone.parent is None:
        return bone.matrix_local.to_quaternion()
    return (bone.parent.matrix_local.inverted() @
            bone.matrix_local).to_quaternion()


def _bone_world_rotation_at(armature, bone, curves, cache, frame):
    """按曲线逐级累加骨骼世界旋转，结果按骨骼和帧缓存。"""
    key = (bone.name, frame)
    cached = cache.get(key)
    if cached is not None:
        return cached
    chain = []
    current = bone
    while current is not None:
        chain.append(current)
        current = current.parent
    rotation = armature.matrix_world.to_quaternion()
    for item in reversed(chain):
        path = f'pose.bones["{item.name}"].rotation_quaternion'
        quaternion = (_quaternion_at(curves, path, frame)
                      if (path, 0) in curves else Quaternion())
        rotation = rotation @ _rest_relative_quaternion(item) @ quaternion
    cache[key] = rotation
    return rotation


def _fix_finger_chain(action, armature, curves, cache, names, axis,
                      compensate):
    """修复一条手指链：末端两节只保留 axis 轴旋转，指定指节补偿头连线。

    末端两节按局部欧拉去掉其余轴的旋转；compensate 指定的指节绕世界轴
    旋转，使「第一指节 head → 第三指节 head」的连线方向与修复前一致；
    compensate 为 -1 时不做补偿。只做曲线求值与四元数运算，不刷新场景。
    """
    bones = [armature.data.bones.get(name) for name in names]
    poses = [armature.pose.bones.get(name) for name in names]
    if any(bone is None or pose is None
           for bone, pose in zip(bones, poses)):
        return False
    base_bone = bones[0]
    parent = base_bone.parent
    if parent is None:
        return False
    paths = [f'pose.bones["{name}"].rotation_quaternion' for name in names]
    present = [bool(_curves(action, path)) for path in paths]
    if not present[1] and not present[2]:
        return False
    times = set()
    for path, has in zip(paths, present):
        if has:
            times |= _key_times(action, path)
    if not times:
        return False
    frames = list(range(int(min(times)), int(max(times)) + 1))
    rest_matrices = [bone.matrix_local for bone in bones]
    rest_quats = [matrix.to_quaternion() for matrix in rest_matrices]
    r01 = rest_quats[0].inverted() @ rest_quats[1]
    head1_offset = (rest_quats[0].inverted() @
                    (rest_matrices[1].to_translation() -
                     rest_matrices[0].to_translation()))
    head2_offset = (rest_quats[1].inverted() @
                    (rest_matrices[2].to_translation() -
                     rest_matrices[1].to_translation()))
    axis_y = Vector((0.0, 1.0, 0.0))
    rest_bend = math.degrees(axis_y.angle((r01 @ axis_y).normalized()))
    base_rest = _rest_relative_quaternion(base_bone)
    values = ([], [], [])
    previous = [None, None, None]
    for frame in frames:
        current = [
            (_quaternion_at(curves, paths[index], frame)
             if present[index] else poses[index].rotation_quaternion.copy())
            for index in range(3)
        ]
        fixed0 = current[0].copy()
        fixed1 = _keep_axis_rotation(current[1], axis)
        fixed2 = _keep_axis_rotation(current[2], axis)
        if compensate >= 0:
            parent_rotation = _bone_world_rotation_at(armature, parent,
                                                      curves, cache, frame)
            base_frame = (parent_rotation @ base_rest).normalized()
            m0 = base_frame @ current[0]
            line_original = m0 @ (head1_offset +
                                  (r01 @ current[1]) @ head2_offset)
            line_fixed = m0 @ (head1_offset +
                               (r01 @ fixed1) @ head2_offset)
            if line_original.length > 1e-9 and line_fixed.length > 1e-9:
                bend = math.degrees(axis_y.angle(
                    ((r01 @ current[1]) @ axis_y).normalized()))
                weight = _compensation_weight(max(bend - rest_bend, 0.0))
                if weight > 1e-6:
                    world_correction = line_fixed.rotation_difference(
                        line_original)
                    if weight < 1.0 - 1e-6:
                        world_correction = Quaternion().slerp(
                            world_correction, weight)
                    if compensate == 0:
                        compensation_frame = base_frame
                        fixed0 = ((compensation_frame.inverted() @
                                   world_correction @ compensation_frame) @
                                  current[0]).normalized()
                    else:
                        compensation_frame = (m0 @ r01).normalized()
                        fixed1 = ((compensation_frame.inverted() @
                                   world_correction @ compensation_frame) @
                                  fixed1).normalized()
        for index, quaternion in enumerate((fixed0, fixed1, fixed2)):
            if previous[index] is not None and quaternion.dot(previous[index]) < 0:
                quaternion.negate()
            previous[index] = quaternion.copy()
            values[index].append(quaternion)
    styles = [_key_styles(curves.get((paths[index], 0))) for index in range(3)]
    for index in range(3):
        if index == compensate or present[index]:
            _write_curves(
                action, paths[index], frames,
                [[q[i] for q in values[index]] for i in range(4)],
                group='finger_fix', styles=styles[index])
    return True


@timeit
def fix_fingers(armature):
    """对两侧所有手指执行修复，返回处理的手指数量。"""
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        return 0
    curves = _curve_map(action)
    cache = {}
    fixed = 0
    for side in ('.L', '.R'):
        for chain in _FINGER_CHAINS:
            names = [name + side for name in chain[:3]]
            if _fix_finger_chain(action, armature, curves, cache, names,
                                 chain[3], chain[4]):
                fixed += 1
    return fixed


_FINGER_BACKUP_PROPS = ('w', 'x', 'y', 'z')
_FINGER_BACKUP_PREFIX = 'flowhmr_finger_q'


def _finger_backup_path(bone_name, component):
    return f'pose.bones["{bone_name}"]["{_FINGER_BACKUP_PREFIX}{component}"]'


def _backup_finger_rotation(action, armature, name):
    """把骨骼旋转曲线备份到自定义属性；没有曲线的骨骼直接跳过。"""
    path = f'pose.bones["{name}"].rotation_quaternion'
    if not _curves(action, path):
        return False
    frames = sorted(_key_times(action, path))
    if not frames:
        return False
    curves = _curve_map(action)
    pose_bone = armature.pose.bones[name]
    columns = _sample_columns(curves, path, frames)
    for component, values in zip(_FINGER_BACKUP_PROPS, columns):
        _write_custom_property_curve(
            pose_bone, _FINGER_BACKUP_PREFIX + component, frames, values)
    return True


def _restore_finger_rotation(action, armature, name):
    """用自定义属性备份恢复骨骼旋转并删除备份曲线。"""
    backup_paths = [_finger_backup_path(name, component)
                    for component in _FINGER_BACKUP_PROPS]
    if not all(_curves(action, path) for path in backup_paths):
        return False
    frames = sorted(set().union(*(_key_times(action, path)
                                  for path in backup_paths)))
    if not frames:
        return False
    curves = _curve_map(action)
    columns = [
        _sample_column(curves, path, frames,
                       default=1.0 if component == 'w' else 0.0)
        for component, path in zip(_FINGER_BACKUP_PROPS, backup_paths)
    ]
    _write_curves(action,
                  f'pose.bones["{name}"].rotation_quaternion',
                  frames, columns, group='finger_fix')
    for path in backup_paths:
        _remove_curves(action, path)
    return True


def apply_finger_fix(armature):
    """备份原始旋转后执行手指修复；缺少的骨骼直接跳过。"""
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        return 0
    for side in ('.L', '.R'):
        for chain in _FINGER_CHAINS:
            for base in chain[:3]:
                name = base + side
                if armature.pose.bones.get(name) is None:
                    continue
                _backup_finger_rotation(action, armature, name)
    return fix_fingers(armature)


def revert_finger_fix(armature):
    """用备份把两侧手指旋转恢复到修复前，返回恢复的骨骼数量。"""
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        return 0
    restored = 0
    for side in ('.L', '.R'):
        for chain in _FINGER_CHAINS:
            for base in chain[:3]:
                name = base + side
                if armature.pose.bones.get(name) is None:
                    continue
                if _restore_finger_rotation(action, armature, name):
                    restored += 1
    return restored


_IK_TARGET_BONES = ('足ＩＫ', 'つま先ＩＫ')
_IK_POSITION_TOLERANCE = 0.005
_IK_ROTATION_TOLERANCE = math.radians(0.5)
_ROTATION_CLEAN_TOLERANCE = math.radians(1.0)
_MMD_HANDLE_START = 20.0 / 127.0
_MMD_HANDLE_END = 107.0 / 127.0


def _simplify_indices(points, frames, tolerance):
    """按时间线性插值后的位置偏差对关键帧做 RDP 简化。"""
    if len(points) < 3:
        return list(range(len(points)))
    keep = {0, len(points) - 1}
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        span = frames[end] - frames[start]
        worst = 0.0
        worst_index = -1
        for index in range(start + 1, end):
            ratio = (frames[index] - frames[start]) / span if span else 0.5
            expected = points[start].lerp(points[end], ratio)
            deviation = (points[index] - expected).length
            if deviation > worst:
                worst = deviation
                worst_index = index
        if worst > tolerance:
            keep.add(worst_index)
            stack.append((start, worst_index))
            stack.append((worst_index, end))
    return sorted(keep)


def _set_straight_bezier(fcurve):
    points = fcurve.keyframe_points
    count = len(points)
    if not count:
        return
    co = np.empty(count * 2, np.float64)
    points.foreach_get('co', co)
    co = co.reshape(-1, 2)
    delta = co[1:] - co[:-1]
    right = co.copy()
    left = co.copy()
    right[:-1] = co[:-1] + delta * _MMD_HANDLE_START
    left[1:] = co[:-1] + delta * _MMD_HANDLE_END
    # 3.6 的 foreach_set 不支持枚举属性，退化为逐点赋值。
    points.foreach_set('handle_left_type', [0] * count)
    points.foreach_set('handle_right_type', [0] * count)
    if points[0].handle_left_type != 'FREE':
        for point in points:
            point.handle_left_type = 'FREE'
            point.handle_right_type = 'FREE'
    points.foreach_set('handle_right', right.ravel())
    points.foreach_set('handle_left', left.ravel())
    points.foreach_set('interpolation', [2] * count)
    fcurve.update()


def _write_bezier_curves(action, path, frames, columns, group='ik_clean'):
    """写入贝塞尔曲线：控制点落在弦上（MMD 的 20/127、107/127），不会过冲。"""
    _remove_curves(action, path)
    for index, values in enumerate(columns):
        fcurve = action.fcurves.new(path, index=index, action_group=group)
        points = fcurve.keyframe_points
        points.add(len(frames))
        for i, (frame, value) in enumerate(zip(frames, values)):
            points[i].co = (float(frame), float(value))
        _set_straight_bezier(fcurve)


def _simplified_quaternion_columns(quaternions, frames):
    kept = _simplify_rotation_indices(quaternions, frames,
                                      _ROTATION_CLEAN_TOLERANCE)
    values = {index: quaternions[index].copy() for index in kept}
    for left, right in zip(kept, kept[1:]):
        first = values[left]
        second = values[right]
        if first.dot(second) < 0.0:
            second = Quaternion((-second.w, -second.x, -second.y, -second.z))
        if first.rotation_difference(second).angle <= _ROTATION_CLEAN_TOLERANCE:
            middle = first.slerp(second, 0.5)
            values[left] = middle.copy()
            values[right] = middle.copy()
    kept_frames = [frames[index] for index in kept]
    columns = [[values[index][axis] for index in kept] for axis in range(4)]
    return kept_frames, columns


def _simplify_rotation_indices(quaternions, frames, tolerance):
    if len(quaternions) < 3:
        return list(range(len(quaternions)))
    keep = {0, len(quaternions) - 1}
    stack = [(0, len(quaternions) - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        span = frames[end] - frames[start]
        worst = 0.0
        worst_index = -1
        for index in range(start + 1, end):
            ratio = (frames[index] - frames[start]) / span if span else 0.5
            expected = quaternions[start].slerp(quaternions[end], ratio)
            deviation = expected.rotation_difference(
                quaternions[index]).angle
            if deviation > worst:
                worst = deviation
                worst_index = index
        if worst > tolerance:
            keep.add(worst_index)
            stack.append((start, worst_index))
            stack.append((worst_index, end))
    return sorted(keep)


def _clean_bone_keys(action, armature, bone_name, stats):
    bone = armature.data.bones.get(bone_name)
    location_path = f'pose.bones["{bone_name}"].location'
    if bone is None or not _curves(action, location_path):
        return 0
    curves = _curve_map(action)
    frames = sorted(_key_times(action, location_path))
    if len(frames) < 2:
        return 0
    local = [
        Vector([_evaluate(curves, location_path, index, frame, 0.0)
                for index in range(3)])
        for frame in frames
    ]
    world_rotation = (armature.matrix_world.to_3x3() @
                      bone.matrix_local.to_3x3())
    world_points = [world_rotation @ value for value in local]
    kept = _simplify_indices(world_points, frames, _IK_POSITION_TOLERANCE)
    positions = {index: world_points[index].copy() for index in kept}
    for left, right in zip(kept, kept[1:]):
        if (positions[right] - positions[left]).length <= _IK_POSITION_TOLERANCE:
            middle = (positions[left] + positions[right]) * 0.5
            positions[left] = middle.copy()
            positions[right] = middle.copy()
    local_rotation = world_rotation.inverted()
    kept_frames = [frames[index] for index in kept]
    columns = [
        [(local_rotation @ positions[index])[axis] for index in kept]
        for axis in range(3)
    ]
    _write_bezier_curves(action, location_path, kept_frames, columns)
    final_frames = set(kept_frames)
    new_curves = _curve_map(action)
    for frame, value in zip(frames, local):
        updated = Vector([_evaluate(new_curves, location_path, index, frame)
                          for index in range(3)])
        delta = (world_rotation @ (updated - value)).length
        stats['position'] = max(stats['position'], delta)

    rotation_path = f'pose.bones["{bone_name}"].rotation_quaternion'
    if _curves(action, rotation_path):
        quaternions = [_quaternion_at(curves, rotation_path, frame)
                       for frame in frames]
        first = quaternions[0]
        deviation = max(first.rotation_difference(value).angle
                        for value in quaternions)
        if deviation <= _IK_ROTATION_TOLERANCE:
            _write_bezier_curves(action, rotation_path,
                                 [frames[0], frames[-1]],
                                 [[first[index]] * 2 for index in range(4)])
            final_frames.update((frames[0], frames[-1]))
            new_curves = _curve_map(action)
            for frame, value in zip(frames, quaternions):
                updated = _quaternion_at(new_curves, rotation_path, frame)
                stats['rotation'] = max(
                    stats['rotation'],
                    value.rotation_difference(updated).angle)
    stats['removed'] += len(set(frames) - final_frames)
    return 1


def _clean_rotation_curves(action, armature, bone_name, stats):
    bone = armature.data.bones.get(bone_name)
    path = f'pose.bones["{bone_name}"].rotation_quaternion'
    if bone is None or not _curves(action, path):
        return 0
    curves = _curve_map(action)
    frames = sorted(_key_times(action, path))
    if len(frames) < 2:
        return 0
    quaternions = [_quaternion_at(curves, path, frame) for frame in frames]
    kept_frames, columns = _simplified_quaternion_columns(quaternions, frames)
    _write_bezier_curves(action, path, kept_frames, columns)
    new_curves = _curve_map(action)
    for frame, value in zip(frames, quaternions):
        updated = _quaternion_at(new_curves, path, frame)
        stats['rotation'] = max(stats['rotation'],
                                value.rotation_difference(updated).angle)
    stats['removed'] += len(set(frames) - set(kept_frames))
    return 1


def clean_ik_keys(armature, stats):
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        return 0
    cleaned = 0
    for side in ('.L', '.R'):
        for base in _IK_TARGET_BONES:
            cleaned += _clean_bone_keys(action, armature, base + side, stats)
    return cleaned


def clean_center_keys(armature, stats):
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        return 0
    return _clean_bone_keys(action, armature, _CENTER_BONE, stats)


def clean_waist_rotation(armature, stats):
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        return 0
    return _clean_rotation_curves(action, armature, _WAIST_BONE, stats)


def clean_foot_rotation(armature, stats):
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        return 0
    return sum(_clean_rotation_curves(action, armature, '足' + side, stats)
               for side in ('.L', '.R'))


def clean_ankle_rotation(armature, stats):
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        return 0
    return sum(_clean_rotation_curves(action, armature, '足首' + side, stats)
               for side in ('.L', '.R'))


def clean_lower_body_rotation(armature, stats):
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        return 0
    return _clean_rotation_curves(action, armature, '下半身', stats)


def _write_custom_bezier(pose_bone, prop_name, frames, values):
    if not _write_custom_property_curve(pose_bone, prop_name, frames, values):
        return False
    action = pose_bone.id_data.animation_data.action
    data_path = f'pose.bones["{pose_bone.name}"]["{prop_name}"]'
    fcurve = next((fc for fc in action.fcurves
                   if fc.data_path == data_path and fc.array_index == 0), None)
    if fcurve is None:
        return False
    _set_straight_bezier(fcurve)
    return True


def _clean_ex_side(action, armature, side, stats):
    name = _EX_BONE + side
    pose_bone = armature.pose.bones.get(name)
    if pose_bone is None:
        return 0
    rotation_path = f'pose.bones["{name}"].rotation_quaternion'
    custom_paths = [_ex_custom_path(side, component)
                    for component in _EX_CUSTOM_PROPS]
    curves = _curve_map(action)
    bone_frames = (sorted(_key_times(action, rotation_path))
                   if _curves(action, rotation_path) else [])
    custom_frames = (sorted(set().union(*(_key_times(action, path)
                                          for path in custom_paths)))
                     if all(_curves(action, path) for path in custom_paths)
                     else [])
    if not bone_frames and not custom_frames:
        return 0
    original = set(bone_frames) | set(custom_frames)
    final = set()
    if bone_frames:
        quaternions = [_quaternion_at(curves, rotation_path, frame)
                       for frame in bone_frames]
        kept_frames, columns = _simplified_quaternion_columns(
            quaternions, bone_frames)
        _write_bezier_curves(action, rotation_path, kept_frames, columns)
        final.update(kept_frames)
        new_curves = _curve_map(action)
        for frame, value in zip(bone_frames, quaternions):
            updated = _quaternion_at(new_curves, rotation_path, frame)
            stats['rotation'] = max(stats['rotation'],
                                    value.rotation_difference(updated).angle)
    if custom_frames:
        stored = [
            Quaternion([
                _evaluate(curves, path, 0, frame,
                          1.0 if component == 'w' else 0.0)
                for component, path in zip(_EX_CUSTOM_PROPS, custom_paths)
            ]).normalized()
            for frame in custom_frames
        ]
        kept_frames, columns = _simplified_quaternion_columns(
            stored, custom_frames)
        for index, path in enumerate(custom_paths):
            prop = f'flowhmr_ex_q{_EX_CUSTOM_PROPS[index]}'
            _write_custom_bezier(pose_bone, prop, kept_frames,
                                 columns[index])
        final.update(kept_frames)
        new_curves = _curve_map(action)
        for frame, value in zip(custom_frames, stored):
            updated = Quaternion([
                _evaluate(new_curves, path, 0, frame,
                          1.0 if component == 'w' else 0.0)
                for component, path in zip(_EX_CUSTOM_PROPS, custom_paths)
            ]).normalized()
            stats['rotation'] = max(stats['rotation'],
                                    value.rotation_difference(updated).angle)
    stats['removed'] += len(original - final)
    return 1


def clean_ex_rotation(armature, stats):
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        return 0
    return sum(_clean_ex_side(action, armature, side, stats)
               for side in ('.L', '.R'))


def clean_toe_rotation(armature, stats):
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        return 0
    return sum(_clean_rotation_curves(action, armature, 'つま先' + side, stats)
               for side in ('.L', '.R'))


_BODY_ROTATION_BONES = (
    '上半身', '上半身1', '上半身2', '首', '頭',
    '肩.L', '肩.R', '腕.L', '腕.R', 'ひじ.L', 'ひじ.R',
    '手首.L', '手首.R',
)
_FINGER_CLEAN_BONES = (
    '親指０', '親指１', '親指２',
    '人指１', '人指２', '人指３',
    '中指１', '中指２', '中指３',
    '薬指１', '薬指２', '薬指３',
    '小指１', '小指２', '小指３',
)


def clean_body_rotation(armature, stats):
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        return 0
    cleaned = 0
    for name in _BODY_ROTATION_BONES:
        cleaned += _clean_rotation_curves(action, armature, name, stats)
    return cleaned


def clean_finger_rotation(armature, stats):
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        return 0
    cleaned = 0
    for side in ('.L', '.R'):
        for base in _FINGER_CLEAN_BONES:
            cleaned += _clean_rotation_curves(action, armature,
                                              base + side, stats)
    return cleaned


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


def adjust_foot_spacing(armature, spacing_m):
    """左右足ＩＫ 沿下半身当前朝向的左右方向各向外平移 spacing_m 米。"""
    shift = float(spacing_m)
    if abs(shift) < 1e-9:
        return False
    left_bone = armature.data.bones.get('足ＩＫ.L')
    right_bone = armature.data.bones.get('足ＩＫ.R')
    if left_bone is None or right_bone is None:
        return False
    action = armature.animation_data.action if armature.animation_data else None
    action_curves = _curve_map(action) if action is not None else {}
    rest_lateral = left_bone.head_local - right_bone.head_local
    rest_lateral.z = 0.0
    if rest_lateral.length < 1e-9:
        rest_lateral = Vector((1.0, 0.0, 0.0))
    else:
        rest_lateral.normalize()
    lower = armature.data.bones.get('下半身')
    if lower is not None:
        local_lateral = lower.matrix_local.to_quaternion().inverted() @ rest_lateral
    else:
        local_lateral = None
    cache = {}

    def lateral_at(frame):
        if local_lateral is None:
            lateral = armature.matrix_world.to_quaternion() @ rest_lateral
        else:
            lateral = _bone_world_rotation_at(armature, lower, action_curves,
                                              cache, frame) @ local_lateral
        lateral.z = 0.0
        if lateral.length < 1e-9:
            return Vector((1.0, 0.0, 0.0))
        return lateral.normalized()

    found = False
    for side, sign in (('.L', 1.0), ('.R', -1.0)):
        pose_bone = armature.pose.bones.get(f'足ＩＫ{side}')
        if pose_bone is None:
            continue
        found = True
        rest_inverse = (armature.matrix_world @
                        pose_bone.bone.matrix_local).to_quaternion().inverted()
        path = pose_bone.path_from_id('location')
        channel_curves = ({fc.array_index: fc for fc in action.fcurves
                           if fc.data_path == path}
                          if action is not None else {})
        if not channel_curves:
            frame = bpy.context.scene.frame_current
            pose_bone.location += rest_inverse @ (
                    lateral_at(frame) * (sign * shift))
            continue
        frames = sorted({round(point.co.x) for fcurve in channel_curves.values()
                         for point in fcurve.keyframe_points})
        if not frames:
            frame = bpy.context.scene.frame_current
            pose_bone.location += rest_inverse @ (
                    lateral_at(frame) * (sign * shift))
            continue
        deltas = {frame: rest_inverse @ (lateral_at(frame) * (sign * shift))
                  for frame in frames}
        for index in range(3):
            fcurve = channel_curves.get(index)
            if fcurve is None:
                fcurve = action.fcurves.new(path, index=index,
                                            action_group='standardize')
                fcurve.keyframe_points.add(len(frames))
                base = pose_bone.location[index]
                for point, frame in zip(fcurve.keyframe_points, frames):
                    point.co = (frame, base + deltas[frame][index])
                    point.interpolation = 'LINEAR'
                fcurve.update()
                continue
            for point in fcurve.keyframe_points:
                amount = deltas[round(point.co.x)][index]
                point.co.y += amount
                point.handle_left.y += amount
                point.handle_right.y += amount
            fcurve.update()
    return found


class FLOWHMR2VMD_OT_adjust_foot_spacing(bpy.types.Operator):
    bl_idname = 'flowhmr2vmd.adjust_foot_spacing'
    bl_label = '调整脚间距'
    bl_description = '通过移动IK骨骼调整角色的脚间距。\n当双腿过于靠近、并拢或发生交叉穿模时，可使用本功能进行调整。'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        root, armature = _selected_model(context)
        if armature is None:
            self.report({'ERROR'}, '请选择 MMD 模型')
            return {'CANCELLED'}
        props = getattr(context.scene, 'flowhmr2vmd_properties', None)
        if props is None:
            return {'CANCELLED'}
        spacing = float(props.foot_spacing)
        if abs(spacing) < 1e-9:
            return {'FINISHED'}
        if not adjust_foot_spacing(armature, spacing):
            self.report({'ERROR'}, '骨架缺少 足ＩＫ 骨骼')
            return {'CANCELLED'}
        context.view_layer.update()
        return {'FINISHED'}


class FLOWHMR2VMD_OT_set_toe_rotation(bpy.types.Operator):
    """UI 按钮：只写入枚举属性，实际逻辑由 toe_rotation 的 update 回调处理。"""
    bl_idname = 'flowhmr2vmd.set_toe_rotation'
    bl_label = '脚尖旋转'
    bl_description = '设置脚尖旋转的开启与关闭\n注：虽然动捕数据中包含脚尖旋转的关键帧，但MMD通常不会为脚尖骨骼注册关键帧'
    bl_options = {'INTERNAL'}

    mode: bpy.props.EnumProperty(name='操作', items=_MODE_ITEMS,
                                 default='DISABLED')

    def execute(self, context):
        props = getattr(context.scene, 'flowhmr2vmd_properties', None)
        if props is None:
            return {'CANCELLED'}
        props.toe_rotation = self.mode
        return {'FINISHED'}


class FLOWHMR2VMD_OT_set_finger_fix(bpy.types.Operator):
    """UI 按钮：只写入枚举属性，实际逻辑由 finger_fix 的 update 回调处理。"""
    bl_idname = 'flowhmr2vmd.set_finger_fix'
    bl_label = '手指修复'
    bl_description = '设置手指修复的开启与关闭'
    bl_options = {'INTERNAL'}

    mode: bpy.props.EnumProperty(name='操作', items=_MODE_ITEMS,
                                 default='DISABLED')

    def execute(self, context):
        props = getattr(context.scene, 'flowhmr2vmd_properties', None)
        if props is None:
            return {'CANCELLED'}
        props.finger_fix = self.mode
        return {'FINISHED'}


class FLOWHMR2VMD_OT_clean_ik_keys(bpy.types.Operator):
    bl_idname = 'flowhmr2vmd.clean_ik_keys'
    bl_label = '清理腿部关键帧'
    bl_description = '清理腿部运动相关的冗余关键帧，以减轻腿部抖动，并减小导出后VMD文件的体积。'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        _root, armature = _selected_model(context)
        if armature is None:
            self.report({'ERROR'}, '请选择 MMD 模型')
            return {'CANCELLED'}
        stats = {'removed': 0, 'position': 0.0, 'rotation': 0.0}
        clean_ik_keys(armature, stats)
        clean_center_keys(armature, stats)
        clean_waist_rotation(armature, stats)
        clean_foot_rotation(armature, stats)
        clean_ankle_rotation(armature, stats)
        clean_toe_rotation(armature, stats)
        clean_ex_rotation(armature, stats)
        clean_lower_body_rotation(armature, stats)
        self.report({'INFO'},
                    f'足部减少 {stats["removed"]} 帧，最大位移偏差 '
                    f'{stats["position"] * 1000.0:.2f} mm，最大旋转偏差 '
                    f'{math.degrees(stats["rotation"]):.2f}°')
        return {'FINISHED'}


class FLOWHMR2VMD_OT_clean_body_keys(bpy.types.Operator):
    bl_idname = 'flowhmr2vmd.clean_body_keys'
    bl_label = '清理身体关键帧'
    bl_description = '清理腿部以外身体运动相关的冗余关键帧，以减轻身体抖动，并减小导出后VMD文件的体积。'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        _root, armature = _selected_model(context)
        if armature is None:
            self.report({'ERROR'}, '请选择 MMD 模型')
            return {'CANCELLED'}
        stats = {'removed': 0, 'position': 0.0, 'rotation': 0.0}
        clean_body_rotation(armature, stats)
        clean_finger_rotation(armature, stats)
        self.report({'INFO'},
                    f'身体减少 {stats["removed"]} 帧，最大位移偏差 '
                    f'{stats["position"] * 1000.0:.2f} mm，最大旋转偏差 '
                    f'{math.degrees(stats["rotation"]):.2f}°')
        return {'FINISHED'}


_BOOST_SPECS = {
    'UP': (Vector((0.0, 0.0, 1.0)), 1.0),
    'DOWN': (Vector((0.0, 0.0, 1.0)), -1.0),
    'LEFT': (Vector((1.0, 0.0, 0.0)), 1.0),
    'RIGHT': (Vector((1.0, 0.0, 0.0)), -1.0),
    'FRONT': (Vector((0.0, 1.0, 0.0)), -1.0),
    'BACK': (Vector((0.0, 1.0, 0.0)), 1.0),
    'VERTICAL': (Vector((0.0, 0.0, 1.0)), 0.0),
    'SIDE': (Vector((1.0, 0.0, 0.0)), 0.0),
    'FRONTBACK': (Vector((0.0, 1.0, 0.0)), 0.0),
    'ALL': (None, 0.0),
}
_BOOST_LOCALS = (Vector((1.0, 0.0, 0.0)),
                 Vector((0.0, 1.0, 0.0)),
                 Vector((0.0, 0.0, 1.0)))


def _boost_point(point, mean, factor, allowed):
    for attribute in ('co', 'handle_left', 'handle_right'):
        vector = getattr(point, attribute)
        deviation = vector.y - mean
        if allowed == 0.0 or deviation * allowed > 0.0:
            vector.y = mean + deviation * factor
            setattr(point, attribute, vector)


def _boost_channel_targets(channels, directions, target, side):
    if target is None:
        return [(index, 0.0) for index in channels]
    best_index = max(channels,
                     key=lambda index: abs(directions[index].dot(target)))
    if side == 0.0:
        return [(best_index, 0.0)]
    alignment = directions[best_index].dot(target)
    return [(best_index, (1.0 if alignment >= 0.0 else -1.0) * side)]


def boost_selected_keyframes(armature, axis, factor, margin):
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        return 0
    active = armature.data.bones.active
    if active is None:
        return 0
    pose_bone = armature.pose.bones.get(active.name)
    if pose_bone is None:
        return 0
    if not (getattr(pose_bone, 'select', False) or pose_bone.bone.select):
        return 0
    target, side = _BOOST_SPECS[axis]
    path = f'pose.bones["{pose_bone.name}"].location'
    channels = {fc.array_index: fc for fc in action.fcurves
                if fc.data_path == path}
    if not channels:
        return 0
    selected_frames = set()
    for fcurve in channels.values():
        selected_frames.update(
            round(point.co.x) for point in fcurve.keyframe_points
            if point.select_control_point)
    if not selected_frames:
        return 0
    rest = (armature.matrix_world @ pose_bone.bone.matrix_local).to_quaternion()
    directions = [(rest @ local).normalized() for local in _BOOST_LOCALS]
    boosted = set()
    touched = False
    for index, allowed in _boost_channel_targets(channels, directions,
                                                 target, side):
        fcurve = channels.get(index)
        if fcurve is None:
            continue
        points = fcurve.keyframe_points
        selected_indices = [index for index, point in enumerate(points)
                            if round(point.co.x) in selected_frames]
        if not selected_indices:
            continue
        runs = []
        run_start = selected_indices[0]
        previous = run_start
        for index in selected_indices[1:]:
            if index == previous + 1:
                previous = index
                continue
            runs.append((run_start, previous))
            run_start = previous = index
        runs.append((run_start, previous))
        fallback = sum(point.co.y for point in points) / len(points)
        for run_start, run_end in runs:
            if run_end > run_start:
                # 基准 = 选中段内所有关键帧值的算术平均（总和 ÷ 数量）；单帧时取左右相邻帧均值
                baseline = sum(points[index].co.y
                               for index in range(run_start, run_end + 1)) / (
                                   run_end - run_start + 1)
            else:
                left = points[run_start - 1].co.y if run_start > 0 else None
                right = (points[run_end + 1].co.y
                         if run_end + 1 < len(points) else None)
                if left is None and right is None:
                    baseline = fallback
                elif left is None:
                    baseline = right
                elif right is None:
                    baseline = left
                else:
                    baseline = (left + right) * 0.5
            for index in range(run_start, run_end + 1):
                _boost_point(points[index], baseline, factor, allowed)
                boosted.add(round(points[index].co.x))
        fcurve.update()
        touched = True
    if not touched:
        return 0
    for fcurve in channels.values():
        if margin > 0 and boosted:
            points = fcurve.keyframe_points
            for index in range(len(points) - 1, -1, -1):
                frame = round(points[index].co.x)
                if frame in boosted:
                    continue
                if any(abs(frame - boosted_frame) <= margin
                       for boosted_frame in boosted):
                    points.remove(points[index])
        _set_straight_bezier(fcurve)
    return 1


class FLOWHMR2VMD_OT_boost_bone_motion(bpy.types.Operator):
    bl_idname = 'flowhmr2vmd.boost_bone_motion'
    bl_label = '位移增强'
    bl_description = '增强指定骨骼在指定方向上的位移幅度，使运动变化更加明显。\n使用前需选中目标骨骼，并在时间轴中框选需要处理的关键帧范围。'
    bl_options = {'REGISTER', 'UNDO'}

    margin: bpy.props.IntProperty(
        name='周围帧数',
        description='删除选中关键帧周围该帧数内的其它关键帧',
        default=3,
        min=0,
        max=50,
    )

    def execute(self, context):
        _root, armature = _selected_model(context)
        if armature is None:
            self.report({'ERROR'}, '请选择 MMD 模型')
            return {'CANCELLED'}
        props = getattr(context.scene, 'flowhmr2vmd_properties', None)
        if props is None:
            return {'CANCELLED'}
        count = boost_selected_keyframes(armature, props.boost_axis,
                                         float(props.boost_strength),
                                         int(self.margin))
        if not count:
            self.report({'ERROR'},
                        '请激活并选中要增强的骨骼，并选择其位移关键帧')
            return {'CANCELLED'}
        context.view_layer.update()
        return {'FINISHED'}
