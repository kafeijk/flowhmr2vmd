"""Panel properties and finger-limit constraint state."""
import math

import bpy

from ..core import finger_limits, mmd
from ..core.constants import MAPPING_ITEMS

_LIMIT_MIN = math.radians(-180.0)
_LIMIT_MAX = math.radians(180.0)

FINGER_LIMITS_MODE_ITEMS = (
    ('ENABLED', '启用', '为手指骨骼创建 LIMIT_ROTATION 限制约束'),
    ('DISABLED', '关闭', '删除本插件添加的手指限制约束'),
)


def _active_armature(context):
    obj = context.active_object
    if obj is None or obj not in context.selected_objects:
        selected = list(context.selected_objects)
        obj = selected[0] if selected else obj
    if obj is None:
        return None
    root, armature = mmd.find_root_and_armature(obj)
    return armature if root is not None else None


def _apply(self, context, armature):
    count, missing = finger_limits.apply_finger_limits(armature, self)
    status = f'已创建 {count} 根手指限制约束。'
    if missing:
        status += f'缺少 {len(missing)} 根：' + '、'.join(missing[:4])
        if len(missing) > 4:
            status += '…'
    self.finger_limit_status = status


def _fallback_armature(context):
    """唯一一个已经带本插件约束的骨架，用于未选中对象时继续实时更新。"""
    found = []
    for obj in context.scene.objects:
        if obj.type != 'ARMATURE':
            continue
        for pose_bone in obj.pose.bones:
            if any(constraint.name.startswith(finger_limits.FINGER_CONSTRAINT_NAME)
                   for constraint in pose_bone.constraints):
                found.append(obj)
                break
    return found[0] if len(found) == 1 else None


def _update_limit(self, context):
    if self.finger_limits_mode != 'ENABLED':
        return
    armature = _active_armature(context) or _fallback_armature(context)
    if armature is None:
        return
    _apply(self, context, armature)


def _update_enabled(self, context):
    armature = _active_armature(context)
    if self.finger_limits_mode != 'ENABLED':
        if armature is not None:
            count = finger_limits.remove_finger_limits(armature)
        else:
            count = sum(
                finger_limits.remove_finger_limits(obj)
                for obj in context.scene.objects if obj.type == 'ARMATURE')
        self.finger_limit_status = f'已删除 {count} 根手指限制约束。'
        return
    if armature is None:
        self.finger_limits_mode = 'DISABLED'
        self.finger_limit_status = '请先选中 MMD 模型对象。'
        return
    _apply(self, context, armature)


class RGBMOCAP_Properties(bpy.types.PropertyGroup):
    folder: bpy.props.StringProperty(
        name='NPZ 文件夹',
        description='flowhmr 动作缓存文件夹',
        subtype='DIR_PATH',
    )
    mapping: bpy.props.EnumProperty(
        name='骨骼名称映射',
        description='选择目标模型的骨骼命名方式',
        items=MAPPING_ITEMS,
        default='FALLBACK',
    )

    show_finger_limits: bpy.props.BoolProperty(
        name='展开手指骨骼约束',
        description='展开后可在下方修改六组基础手指的 XYZ 最小/最大角度',
        default=False,
    )
    finger_limits_mode: bpy.props.EnumProperty(
        name='手指约束',
        description='启用时创建 限定旋转 约束，关闭时删除本插件添加的约束',
        items=FINGER_LIMITS_MODE_ITEMS,
        default='DISABLED',
        update=_update_enabled,
    )
    finger_limit_status: bpy.props.StringProperty(
        name='手指约束状态',
        default='',
    )

    thumb0_x_min: bpy.props.FloatProperty(name='拇指0 X 最小', default=math.radians(-30.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    thumb0_x_max: bpy.props.FloatProperty(name='拇指0 X 最大', default=math.radians(60.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    thumb0_y_min: bpy.props.FloatProperty(name='拇指0 Y 最小', default=math.radians(-30.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    thumb0_y_max: bpy.props.FloatProperty(name='拇指0 Y 最大', default=math.radians(60.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    thumb0_z_min: bpy.props.FloatProperty(name='拇指0 Z 最小', default=math.radians(-25.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    thumb0_z_max: bpy.props.FloatProperty(name='拇指0 Z 最大', default=math.radians(35.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')

    thumb1_x_min: bpy.props.FloatProperty(name='拇指1 X 最小', default=math.radians(-30.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    thumb1_x_max: bpy.props.FloatProperty(name='拇指1 X 最大', default=math.radians(60.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    thumb1_y_min: bpy.props.FloatProperty(name='拇指1 Y 最小', default=math.radians(-20.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    thumb1_y_max: bpy.props.FloatProperty(name='拇指1 Y 最大', default=math.radians(70.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    thumb1_z_min: bpy.props.FloatProperty(name='拇指1 Z 最小', default=math.radians(-50.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    thumb1_z_max: bpy.props.FloatProperty(name='拇指1 Z 最大', default=math.radians(65.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')

    thumb2_x_min: bpy.props.FloatProperty(name='拇指2 X 最小', default=math.radians(-15.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    thumb2_x_max: bpy.props.FloatProperty(name='拇指2 X 最大', default=math.radians(30.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    thumb2_y_min: bpy.props.FloatProperty(name='拇指2 Y 最小', default=math.radians(0.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    thumb2_y_max: bpy.props.FloatProperty(name='拇指2 Y 最大', default=math.radians(0.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    thumb2_z_min: bpy.props.FloatProperty(name='拇指2 Z 最小', default=math.radians(0.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    thumb2_z_max: bpy.props.FloatProperty(name='拇指2 Z 最大', default=math.radians(0.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')

    index1_x_min: bpy.props.FloatProperty(name='食指1 X 最小', default=math.radians(-30.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    index1_x_max: bpy.props.FloatProperty(name='食指1 X 最大', default=math.radians(90.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    index1_y_min: bpy.props.FloatProperty(name='食指1 Y 最小', default=math.radians(-12.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    index1_y_max: bpy.props.FloatProperty(name='食指1 Y 最大', default=math.radians(6.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    index1_z_min: bpy.props.FloatProperty(name='食指1 Z 最小', default=math.radians(-15.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    index1_z_max: bpy.props.FloatProperty(name='食指1 Z 最大', default=math.radians(30.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')

    index2_x_min: bpy.props.FloatProperty(name='食指2 X 最小', default=math.radians(-25.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    index2_x_max: bpy.props.FloatProperty(name='食指2 X 最大', default=math.radians(140.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    index2_y_min: bpy.props.FloatProperty(name='食指2 Y 最小', default=math.radians(0.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    index2_y_max: bpy.props.FloatProperty(name='食指2 Y 最大', default=math.radians(0.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    index2_z_min: bpy.props.FloatProperty(name='食指2 Z 最小', default=math.radians(0.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    index2_z_max: bpy.props.FloatProperty(name='食指2 Z 最大', default=math.radians(0.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')

    index3_x_min: bpy.props.FloatProperty(name='食指3 X 最小', default=math.radians(-20.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    index3_x_max: bpy.props.FloatProperty(name='食指3 X 最大', default=math.radians(85.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    index3_y_min: bpy.props.FloatProperty(name='食指3 Y 最小', default=math.radians(0.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    index3_y_max: bpy.props.FloatProperty(name='食指3 Y 最大', default=math.radians(0.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    index3_z_min: bpy.props.FloatProperty(name='食指3 Z 最小', default=math.radians(0.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    index3_z_max: bpy.props.FloatProperty(name='食指3 Z 最大', default=math.radians(0.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')

    @staticmethod
    def register():
        if not hasattr(bpy.types.Scene, 'rgbmocap_properties'):
            bpy.types.Scene.rgbmocap_properties = bpy.props.PointerProperty(
                type=RGBMOCAP_Properties)

    @staticmethod
    def unregister():
        if hasattr(bpy.types.Scene, 'rgbmocap_properties'):
            del bpy.types.Scene.rgbmocap_properties


def register():
    RGBMOCAP_Properties.register()


def unregister():
    RGBMOCAP_Properties.unregister()
