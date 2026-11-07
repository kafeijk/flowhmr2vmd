"""Finger rotation limits for MMD humanoid hands.

The six base ranges below are exposed in the panel in degrees; the Blender
angle properties store radians and the constraints consume radians directly.
中指 / 薬指 / 小指 reuse the corresponding 人指 ranges; .R mirrors X and Y
(swap and negate) while Z stays unchanged.
"""
import bpy

FINGER_CONSTRAINT_NAME = '手指旋转限制'

BASE_NAMES = (
    ('親指０', 'thumb0'),
    ('親指１', 'thumb1'),
    ('親指２', 'thumb2'),
    ('人指１', 'index1'),
    ('人指２', 'index2'),
    ('人指３', 'index3'),
)

# UI 显示名：日文骨骼名保持内部使用，面板只显示“左 + 中文手指名 + 半角数字”
FINGER_GROUPS = [
    ('左拇指0', 'thumb0'),
    ('左拇指1', 'thumb1'),
    ('左拇指2', 'thumb2'),
    ('左食指1', 'index1'),
    ('左食指2', 'index2'),
    ('左食指3', 'index3'),
]


def property_names():
    """面板上的 36 个角度属性名。"""
    return [f'{prefix}_{axis}_{bound}'
            for _, prefix in BASE_NAMES
            for axis in ('x', 'y', 'z')
            for bound in ('min', 'max')]


DIGITS = ('１', '２', '３')
REUSED_FINGERS = ('中指', '薬指', '小指')


def base_limits(props):
    limits = {}
    for name, prefix in BASE_NAMES:
        limits[name] = tuple(
            (float(getattr(props, f'{prefix}_{axis}_min')),
             float(getattr(props, f'{prefix}_{axis}_max')))
            for axis in ('x', 'y', 'z'))
    return limits


def all_limits(props):
    limits = base_limits(props)
    for finger in REUSED_FINGERS:
        for digit in DIGITS:
            limits[f'{finger}{digit}'] = limits[f'人指{digit}']
    return limits


def mirrored(limits):
    x, y, z = limits
    return ((-x[1], -x[0]), (-y[1], -y[0]), z)


def limits_for(name, limits):
    if name.endswith('.L'):
        return limits.get(name[:-2])
    if name.endswith('.R'):
        value = limits.get(name[:-2])
        return mirrored(value) if value is not None else None
    return None


def _owned(constraint):
    return (constraint.name == FINGER_CONSTRAINT_NAME
            or constraint.name.startswith(FINGER_CONSTRAINT_NAME + '.'))


def apply_finger_limits(armature, props):
    limits = all_limits(props)
    count = 0
    missing = []
    for base in limits:
        for suffix in ('.L', '.R'):
            bone_name = base + suffix
            pose_bone = armature.pose.bones.get(bone_name)
            if pose_bone is None:
                missing.append(bone_name)
                continue
            value = limits_for(bone_name, limits)
            existing = [c for c in pose_bone.constraints if _owned(c)]
            constraint = existing[0] if existing else None
            for extra in existing[1:]:
                pose_bone.constraints.remove(extra)
            if constraint is not None and constraint.type != 'LIMIT_ROTATION':
                pose_bone.constraints.remove(constraint)
                constraint = None
            if constraint is None:
                constraint = pose_bone.constraints.new('LIMIT_ROTATION')
                constraint.name = FINGER_CONSTRAINT_NAME
            constraint.owner_space = 'LOCAL'
            constraint.mute = False
            for axis, (minimum, maximum) in zip('xyz', value):
                low, high = sorted((minimum, maximum))
                setattr(constraint, f'use_limit_{axis}', True)
                setattr(constraint, f'min_{axis}', low)
                setattr(constraint, f'max_{axis}', high)
            count += 1
    return count, missing


def remove_finger_limits(armature):
    count = 0
    for pose_bone in armature.pose.bones:
        for constraint in list(pose_bone.constraints):
            if _owned(constraint):
                pose_bone.constraints.remove(constraint)
                count += 1
    return count
