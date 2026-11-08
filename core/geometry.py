"""Contact geometry extraction from the imported PMX meshes.

Adapted from RGB MoCap's mocap/retarget/blender_geometry.py (MIT).
"""
import bpy
import numpy as np

from .constants import A


class GeometryError(Exception):
    pass


def _bone_map(armature, mapping):
    mapped = {}
    for index, name in enumerate(mapping):
        mapped.setdefault(name, index)
    for side, suffix in enumerate(('.L', '.R')):
        for stem, joint in (('足D', 1 + side), ('ひざD', 4 + side),
                             ('足首D', 7 + side), ('足先EX', 10 + side)):
            mapped.setdefault(stem + suffix, joint)
    result = {}
    for bone in armature.data.bones:
        parent = bone
        while parent is not None and parent.name not in mapped:
            parent = parent.parent
        result[bone.name] = mapped[parent.name] if parent is not None else 0
    return result, mapped


def build_contact_geometry(armature, meshes, mapping, rest):
    bone_map, mapped = _bone_map(armature, mapping)
    vertices = []
    weights = []
    wrist_weights = []
    for obj in meshes:
        matrix = np.array(obj.matrix_world)
        count = len(obj.data.vertices)
        co = np.empty(count * 3, np.float32)
        obj.data.vertices.foreach_get('co', co)
        co = co.reshape(-1, 3)
        world = (co @ matrix[:3, :3].T + matrix[:3, 3]) @ A
        vertices.append(world)
        # 预先把“顶点组序号 → 关节槽位”建成整数表，顶点循环里只做
        # 整数索引，避免每个顶点组都取名字再做字符串查表。
        joint_by_group = np.full(len(obj.vertex_groups), -1, np.int64)
        wrist_by_group = np.full(len(obj.vertex_groups), -1, np.int64)
        for group in obj.vertex_groups:
            joint = bone_map.get(group.name)
            if joint is not None:
                joint_by_group[group.index] = joint
            for side in (0, 1):
                if group.name == mapping[20 + side]:
                    wrist_by_group[group.index] = side
        weight = np.zeros((count, 22))
        wrist = np.zeros((count, 2))
        for vertex in obj.data.vertices:
            index = vertex.index
            for item in vertex.groups:
                joint = joint_by_group[item.group]
                if joint >= 0:
                    weight[index, joint] += item.weight
                side = wrist_by_group[item.group]
                if side >= 0:
                    wrist[index, side] += item.weight
        total = weight.sum(1)
        missing = total < 1e-8
        weight[missing, 0] = 1.0
        weight /= np.maximum(weight.sum(1, keepdims=True), 1e-8)
        weights.append(weight)
        wrist_weights.append(wrist)

    if not vertices:
        raise GeometryError('模型没有网格，无法计算脚底接触。')
    v = np.concatenate(vertices)
    w = np.concatenate(weights)
    hw = np.concatenate(wrist_weights)
    height = float(rest[:, 1].max() - rest[:, 1].min())
    payload = {}

    for side in (0, 1):
        foot = 7 + side
        toe = 10 + side
        shin = 4 + side
        p = rest[foot]
        length = float(np.linalg.norm(rest[toe] - p))
        mask = ((w[:, [foot, toe, shin]].sum(1) > .65) &
                (v[:, 1] < p[1] + .025 * height) &
                (np.abs(v[:, 0] - p[0]) < max(.10 * height, .12)) &
                (np.abs(v[:, 2] - p[2]) < max(2.5 * length, .22)))
        ids = np.flatnonzero(mask)
        if len(ids) <= 10:
            raise GeometryError(f'第 {side + 1} 只脚的顶点权重不足，无法计算脚底接触。')
        ground = float(np.percentile(v[ids, 1], 1))
        bottom = ids[v[ids, 1] < ground + max(.006, .004 * height)]
        if len(bottom) < 4:
            raise GeometryError(f'第 {side + 1} 只脚的有效鞋底顶点不足。')
        forward = (rest[toe] - p) * np.array([1., 0., 1.])
        forward /= max(float(np.linalg.norm(forward)), 1e-8)
        along = (v[bottom] - p) @ forward
        split = float(np.median(along))
        heel = bottom[along <= split]
        fore = bottom[along > split]
        if len(heel) < 2 or len(fore) < 2:
            heel = bottom[:len(bottom) // 2]
            fore = bottom[len(bottom) // 2:]
        lookup = {int(value): index for index, value in enumerate(ids)}
        payload.update({
            f'foot_v{side}': v[ids], f'foot_w{side}': w[ids],
            f'foot_ids{side}': ids,
            f'heel{side}': np.array([lookup[int(i)] for i in heel], np.int64),
            f'toe{side}': np.array([lookup[int(i)] for i in fore], np.int64),
        })

        wrist_index = 20 + side
        wrist_name = mapping[wrist_index]
        fingers = [b for b in armature.data.bones
                   if b.name.startswith(wrist_name) and b.name != wrist_name]
        lengths = [np.linalg.norm((np.array(armature.matrix_world @ b.head_local) @ A) - rest[wrist_index])
                   for b in fingers]
        lengths = [x for x in lengths if x > .015]
        palm_length = max(.025, float(np.median(lengths))) if lengths else .055
        hand_dist = np.linalg.norm(v - rest[wrist_index], axis=1)
        hand_ids = np.flatnonzero(
            (hw[:, side] > .95) & (hand_dist > palm_length * .15) &
            (hand_dist < min(palm_length * 1.1, .14 * height)))
        if len(hand_ids) < 8:
            hand_ids = np.flatnonzero((hw[:, side] > .85) & (hand_dist < .10 * height))
        if len(hand_ids) <= 3:
            raise GeometryError(f'第 {side + 1} 只手的掌心顶点权重不足。')
        payload.update({
            f'palm_v{side}': v[hand_ids], f'palm_w{side}': w[hand_ids],
            f'palm_ids{side}': hand_ids,
        })

    return payload
