"""Finger rotation limits for MMD humanoid hands.

The six base ranges below are exposed in the panel in degrees; the Blender
angle properties store radians and the constraints consume radians directly.
中指 / 薬指 / 小指 reuse the corresponding 人指 ranges; .R mirrors X and Y
(swap and negate) while Z stays unchanged.
"""
import math

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


# 中指 / 薬指第一节互相防交叉，左侧与右侧各一对（双向）。
PAIRED_FINGERS = (
    ('中指１.L', '薬指１.L', 'index1'),
    ('薬指１.L', '中指１.L', 'index1'),
    ('中指１.R', '薬指１.R', 'index1'),
    ('薬指１.R', '中指１.R', 'index1'),
)

DRIVER_LIMIT = math.radians(15.0)


def apply_finger_drivers(armature, props, scene):
    """让中指与薬指第一节的 Z 角度差保持在 ±15° 内。

    本骨骼的上下限都跟着另一根手指的当前角度走，因此既不会越过对方
    造成交叉，也不会向两侧张开成明显的 V 形；最终范围再受面板值限制。
    """
    count = 0
    for target, source, prefix in PAIRED_FINGERS:
        target_bone = armature.pose.bones.get(target)
        source_bone = armature.pose.bones.get(source)
        if target_bone is None or source_bone is None:
            continue
        constraint = target_bone.constraints.get(FINGER_CONSTRAINT_NAME)
        if constraint is None:
            continue
        for attribute in ('min_z', 'max_z'):
            constraint.driver_remove(attribute)
        for attribute, bound, offset in (('min_z', 'z_min', -DRIVER_LIMIT),
                                         ('max_z', 'z_max', DRIVER_LIMIT)):
            fcurve = constraint.driver_add(attribute)
            driver = fcurve.driver
            driver.type = 'SCRIPTED'
            for name, component in (('qw', 0), ('qz', 3)):
                variable = driver.variables.new()
                variable.name = name
                variable.type = 'SINGLE_PROP'
                variable.targets[0].id_type = 'OBJECT'
                variable.targets[0].id = armature
                variable.targets[0].data_path = (
                    f'pose.bones["{source}"].rotation_quaternion[{component}]')
            for name, bound_name in (('lo', f'{prefix}_z_min'),
                                     ('hi', f'{prefix}_z_max')):
                variable = driver.variables.new()
                variable.name = name
                variable.type = 'SINGLE_PROP'
                variable.targets[0].id_type = 'SCENE'
                variable.targets[0].id = scene
                variable.targets[0].data_path = props.path_from_id(bound_name)
            angle = '2*atan2(qz, abs(qw))'
            if offset < 0:
                driver.expression = f'max(lo, ({angle} - {DRIVER_LIMIT}))'
            else:
                driver.expression = f'min(hi, ({angle} + {DRIVER_LIMIT}))'
            count += 1
    return count


def remove_finger_drivers(armature):
    count = 0
    for target, _source, _prefix in PAIRED_FINGERS:
        bone = armature.pose.bones.get(target)
        if bone is None:
            continue
        constraint = bone.constraints.get(FINGER_CONSTRAINT_NAME)
        if constraint is None:
            continue
        for attribute in ('min_z', 'max_z'):
            if constraint.driver_remove(attribute):
                count += 1
    return count
