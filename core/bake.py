"""Key the solved stance directly on the selected MMD armature.

Adapted from RGB MoCap's mocap/retarget/blender_bake.py (MIT).
"""
import bpy
import numpy as np
from mathutils import Matrix

from .constants import A


def rotation(matrix):
    u, _, v = np.linalg.svd(matrix)
    result = u @ v
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1
        result = u @ v
    return result


def align(a, b):
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    v = np.cross(a, b)
    c = float(np.clip(a @ b, -1, 1))
    if c < -0.999999:
        axis = np.cross(a, np.eye(3)[np.argmin(np.abs(a))])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    skew = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])
    return np.eye(3) + skew + skew @ skew / max(1 + c, 1e-10)


def fcurves(obj):
    action = obj.animation_data.action
    curves = getattr(action, 'fcurves', None)
    if curves is not None and len(curves):
        return curves
    try:
        from bpy_extras import anim_utils
        slot = obj.animation_data.action_slot
        if slot is not None:
            bag = anim_utils.action_get_channelbag_for_slot(action, slot)
            if bag is not None:
                return bag.fcurves
    except Exception:
        pass
    return action.fcurves


def key_values(obj, path, values):
    count = len(values)
    for index in range(values.shape[1]):
        curve = next((c for c in fcurves(obj)
                      if c.data_path == path and c.array_index == index), None)
        if curve is None:
            curve = fcurves(obj).new(data_path=path, index=index)
        curve.keyframe_points.clear()
        curve.keyframe_points.add(count)
        curve.keyframe_points.foreach_set(
            'co', np.column_stack((np.arange(1, count + 1), values[:, index])).reshape(-1))
        for point in curve.keyframe_points:
            point.interpolation = 'LINEAR'
        curve.update()


def body_trajectory_root(parents, mapped):
    roots = set()
    for name in mapped:
        seen = set()
        current = name
        while True:
            if current not in parents or current in seen:
                raise ValueError('骨骼层级无效。')
            seen.add(current)
            parent = parents[current]
            if parent is None:
                roots.add(current)
                break
            current = parent
    if len(roots) != 1:
        raise ValueError('映射骨骼不在同一层级根下。')
    return roots.pop()


def _finger_extra(armature, stance, source):
    frames = source['frames52']
    rest = source['rest52']
    ends = source['ends52']
    names = source['fnames']
    bone_indices = {b.name: i for i, b in enumerate(armature.data.bones)}
    rest_bl = np.array([armature.matrix_world @ b.head_local
                        for b in armature.data.bones])
    extra = {}
    for joint, name in enumerate(names[22:], 22):
        side = 'left' if name.startswith('Left') else 'right'
        hand = 20 if side == 'left' else 21
        suffix = name.split('Hand')[1]
        finger = suffix[:-1]
        phalanx = int(suffix[-1])
        jp = {'Thumb': '親指', 'Index': '人指', 'Middle': '中指',
              'Ring': '薬指', 'Pinky': '小指'}[finger]
        number = phalanx - 1 if finger == 'Thumb' else phalanx
        target = jp + '０１２３'[number] + ('.L' if side == 'left' else '.R')
        if target not in bone_indices:
            continue
        bone = armature.data.bones[target]
        vtarget = (np.array(armature.matrix_world @ bone.tail_local) -
                   rest_bl[bone_indices[target]])
        vsource = rest[joint + 1] - rest[joint] if phalanx < 3 else ends[joint]
        correction = align(vtarget @ A, vsource)
        ch = frames[0, hand, :3, :3].T @ stance['world'][0, hand]
        relative = frames[:, hand, :3, :3].swapaxes(-1, -2) @ frames[:, joint, :3, :3]
        extra[target] = stance['world'][:, hand] @ ch.T @ relative @ correction
    return extra


def bake_solution(armature, mapping, stance, source, skip_indices=()):
    n = len(stance['root'])
    skipped = set(skip_indices)
    mapped = {}
    for index, name in enumerate(mapping):
        if index in skipped:
            continue
        mapped.setdefault(name, index)
    keyed_body = set(mapped.keys())
    extra = _finger_extra(armature, stance, source)
    keyed_rotation = set(keyed_body) | set(extra)
    if '腰' in armature.data.bones:
        keyed_rotation.add('腰')

    parent_map = {b.name: (b.parent.name if b.parent else None)
                  for b in armature.data.bones}
    try:
        root_name = body_trajectory_root(parent_map, mapped)
    except ValueError:
        root_name = None
    if root_name is None:
        raise ValueError('无法确定动作根骨骼。')

    needed = set(keyed_rotation) | {root_name}
    for name in list(keyed_rotation) + [root_name]:
        parent = parent_map.get(name)
        while parent:
            needed.add(parent)
            parent = parent_map.get(parent)

    depth = {}
    for bone in armature.data.bones:
        value = 0
        parent = bone.parent
        while parent:
            value += 1
            parent = parent.parent
        depth[bone.name] = value
    bones = [b for b in armature.data.bones
             if b.name in needed and b.name in depth]
    bones.sort(key=lambda b: depth[b.name])
    indices = {b.name: i for i, b in enumerate(bones)}
    parents = np.array([indices[b.parent.name] if b.parent else -1 for b in bones])
    rest_bl = np.array([armature.matrix_world @ b.head_local for b in bones])
    rest_rotation = np.array([rotation(np.array(armature.matrix_world @ b.matrix_local)[:3, :3])
                              for b in bones])

    world_override = {}
    for name in keyed_body:
        world_override[name] = stance['world'][:, mapped[name]]
    if '腰' in keyed_rotation:
        world_override['腰'] = stance['world'][:, 0]
    for name, value in extra.items():
        world_override[name] = value

    rot = np.zeros((n, len(bones), 3, 3))
    positions = np.zeros((n, len(bones), 3))
    quaternions = {}
    for j, bone in enumerate(bones):
        parent = parents[j]
        parent_rotation = (np.tile(np.eye(3), (n, 1, 1)) if parent < 0
                           else rot[:, parent])
        before = (rest_rotation[j] if parent < 0
                  else rest_rotation[parent].T @ rest_rotation[j])
        previous = parent_rotation @ before
        if bone.name in world_override:
            rot[:, j] = A @ world_override[bone.name] @ A.T @ rest_rotation[j]
        else:
            rot[:, j] = previous
        if parent < 0:
            positions[:, j] = rest_bl[j]
        else:
            positions[:, j] = (positions[:, parent] +
                               np.einsum('tij,j->ti',
                                         rot[:, parent] @ rest_rotation[parent].T,
                                         rest_bl[j] - rest_bl[parent]))
        if bone.name in keyed_rotation:
            basis = previous.swapaxes(-1, -2) @ rot[:, j]
            q = np.array([tuple(Matrix(m.tolist()).to_quaternion().normalized())
                          for m in basis])
            for t in range(1, n):
                if q[t] @ q[t - 1] < 0:
                    q[t] *= -1
            quaternions[bone.name] = q

    root_index = indices[mapping[0]]
    shift = stance['root'] @ A.T - positions[:, root_index]
    trajectory_bone = armature.data.bones[root_name]
    rest_to_world = np.array(armature.matrix_world @ trajectory_bone.matrix_local)[:3, :3]
    root_location = shift @ np.linalg.inv(rest_to_world).T

    # MMD IK and waist-cancel grants must not override the baked FK in the viewport.
    muted = []
    for name in set(keyed_rotation) | {b.name for b in armature.pose.bones
                                       if b.name.startswith('腰キャンセル')}:
        pose_bone = armature.pose.bones.get(name)
        if pose_bone is None:
            continue
        for constraint in pose_bone.constraints:
            if not constraint.mute:
                constraint.mute = True
                muted.append(name)

    armature.animation_data_clear()
    armature.animation_data_create()
    action = None
    for name, q in quaternions.items():
        if q is None:
            continue
        pose_bone = armature.pose.bones.get(name)
        if pose_bone is None:
            continue
        pose_bone.rotation_mode = 'QUATERNION'
        pose_bone.rotation_quaternion = q[0]
        pose_bone.keyframe_insert('rotation_quaternion', frame=1)
        if action is None:
            action = armature.animation_data.action
            action.name = 'FlowHMR2VMD'
        key_values(armature, pose_bone.path_from_id('rotation_quaternion'), q)

    root_pose = armature.pose.bones.get(root_name)
    if root_pose is not None:
        root_pose.location = root_location[0]
        root_pose.keyframe_insert('location', frame=1)
        if action is None:
            action = armature.animation_data.action
            action.name = 'FlowHMR2VMD'
        key_values(armature, root_pose.path_from_id('location'), root_location)
    if action is None:
        raise ValueError('没有可写入的骨骼通道。')

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = n
    scene.render.fps = round(source['fps'])
    scene.render.fps_base = round(source['fps']) / source['fps']
    scene.frame_set(1)
    return dict(action=action.name, frames=n, bones=len(quaternions),
                muted=muted,
                skipped_toe=any(index in skipped for index in (10, 11)))
