"""MMD object discovery and skeleton validation.

Independent implementation: only depends on MMD Tools' ``mmd_type``
property, no other add-ons are imported.

Portions adapted from RGB MoCap (MIT License).
"""
import bpy


def mmd_tools_enabled():
    return hasattr(bpy.types.Object, 'mmd_type')


def find_pmx_root_with_child(child):
    """根据child寻找pmx对应空物体"""
    if not child:
        return None

    parent = child
    while parent:
        if getattr(parent, 'mmd_type', None) == 'ROOT':
            return parent
        parent = parent.parent

    return None


def find_armature(root):
    return next((o for o in root.children if o.type == 'ARMATURE'), None)


def find_root_and_armature(obj):
    if obj is None or not mmd_tools_enabled():
        return None, None
    root = find_pmx_root_with_child(obj)
    if root is None:
        return None, None
    return root, find_armature(root)


def find_pmx_objects(armature):
    return list(child for child in armature.children if child.type == 'MESH' and child.mmd_type == 'NONE')


def trajectory_root(armature, mapping):
    """Common topmost root of all mapped bones."""
    parents = {b.name: (b.parent.name if b.parent else None)
               for b in armature.data.bones}
    unique = [name for i, name in enumerate(mapping)
              if name not in mapping[:i] and name in parents]
    roots = set()
    for name in unique:
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


def validate_mapping(armature, mapping):
    """Return a Chinese error message, or None when the skeleton is usable."""
    bones = set(armature.data.bones.keys())
    missing = [name for name in mapping if name not in bones]
    if missing:
        return '骨架缺少以下骨骼：' + '、'.join(missing)
    # The fallback intentionally repeats 上半身 at the virtual middle-spine
    # slot. Reject any other duplicate because it would silently drop a channel.
    seen = {}
    for index, name in enumerate(mapping):
        if name in seen and index != 6:
            return f'骨骼映射重复：{name}（第 {seen[name] + 1} 与第 {index + 1} 项）'
        seen.setdefault(name, index)
    try:
        trajectory_root(armature, mapping)
    except ValueError as exc:
        return str(exc)
    if not find_pmx_objects(armature):
        return '骨架下没有蒙皮网格，无法计算脚底接触。'
    return None


def toe_is_granted(armature, mapping):
    """True when 足先EX sits under the hidden granted 足首D chain."""
    if mapping[10] != '足先EX.L':
        return False
    bone = armature.data.bones.get(mapping[10])
    if bone is None or bone.parent is None:
        return False
    parent_name = bone.parent.name
    for suffix in ('.L', '.R'):
        if parent_name.endswith(suffix):
            parent_name = parent_name[:-len(suffix)]
            break
    if not parent_name.endswith('D'):
        return False
    pose_bone = armature.pose.bones.get(mapping[10])
    if pose_bone is None or pose_bone.parent is None:
        return False
    return any(getattr(c, 'type', '') == 'TRANSFORM' and
               getattr(c, 'name', '') == 'mmd_additional_rotation'
               for c in pose_bone.parent.constraints)
