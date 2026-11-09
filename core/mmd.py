"""MMD object discovery and skeleton validation.

Independent implementation: only depends on MMD Tools' ``mmd_type``
property, no other add-ons are imported.

Portions adapted from RGB MoCap (MIT License).
"""
import bpy


def mmd_tools_enabled():
    return hasattr(bpy.types.Object, 'mmd_type')


def _find_root(child):
    parent = child
    while parent:
        if getattr(parent, 'mmd_type', None) == 'ROOT':
            return parent
        parent = parent.parent
    return None


def _find_armature(root):
    return next((o for o in root.children if o.type == 'ARMATURE'), None)


def find_root_and_armature(obj):
    if obj is None or not mmd_tools_enabled():
        return None, None
    root = _find_root(obj)
    if root is None:
        return None, None
    return root, _find_armature(root)


STATE_ROTATE_TOE = 'flowhmr2vmd_rotate_toe'
STATE_FINGER_LIMITS = 'flowhmr2vmd_finger_limits'
STATE_FINGER_DRIVER = 'flowhmr2vmd_finger_driver'


def read_state(root):
    """根对象上记录的开关状态；没有对应属性时默认关闭。"""
    if root is None:
        return {key: False for key in
                (STATE_ROTATE_TOE, STATE_FINGER_LIMITS, STATE_FINGER_DRIVER)}
    return {key: bool(root.get(key, False)) for key in
            (STATE_ROTATE_TOE, STATE_FINGER_LIMITS, STATE_FINGER_DRIVER)}


def write_state(root, key, value):
    """开启时写入 True，关闭时移除该属性。"""
    if root is None:
        return
    if value:
        root[key] = True
    elif key in root:
        del root[key]


def meshes_for(armature):
    meshes = []
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        try:
            if obj.find_armature() == armature:
                meshes.append(obj)
        except Exception:
            continue
    if meshes:
        return meshes
    # Fallback for unusual hierarchies: walk the armature's children.
    stack = list(armature.children)
    while stack:
        obj = stack.pop()
        if obj.type == 'MESH':
            meshes.append(obj)
        stack.extend(obj.children)
    return meshes


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
    if not meshes_for(armature):
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
