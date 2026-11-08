"""Panel properties and finger-limit constraint state."""
import math

import bpy

from ..core import finger_limits, mmd
from ..operators.bone_operators import convert_toe_rotation

_LIMIT_MIN = math.radians(-180.0)
_LIMIT_MAX = math.radians(180.0)

FINGER_LIMITS_MODE_ITEMS = (
    ('ENABLED', '启用', '为手指骨骼创建 LIMIT_ROTATION 限制约束'),
    ('DISABLED', '关闭', '删除本插件添加的手指限制约束'),
)

FINGER_DRIVER_MODE_ITEMS = (
    ('ENABLED', '启用', '创建中指 / 无名指约束驱动器'),
    ('DISABLED', '关闭', '删除约束驱动器并把约束恢复为面板范围'),
)

ROTATE_TOE_MODE_ITEMS = (
    ('ENABLED', '开启', '把 つま先 的旋转转换到 足先EX'),
    ('DISABLED', '关闭', '把 足先EX 的旋转转换回 つま先'),
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
    finger_limits.apply_finger_limits(armature, self)


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


_ROTATE_TOE_BUSY = False


def _report_error(context, message):
    try:
        window_manager = context.window_manager

        def draw(menu, _message=message):
            menu.layout.label(text=_message, icon='ERROR')

        window_manager.popup_menu(draw, title='脚尖旋转', icon='ERROR')
    except Exception:
        print('脚尖旋转:', message)


def _update_rotate_toe(self, context):
    """开启：つま先 → 足先EX；关闭：足先EX → つま先。"""
    global _ROTATE_TOE_BUSY
    if _ROTATE_TOE_BUSY:
        return
    armature = _active_armature(context) or _fallback_armature(context)
    if armature is None:
        _report_error(context, '请先选中 MMD 模型')
        return
    _ROTATE_TOE_BUSY = True
    try:
        ok, message = convert_toe_rotation(
            armature, self.rotate_toe_mode == 'ENABLED')
        if ok:
            return
        self.rotate_toe_mode = ('DISABLED' if self.rotate_toe_mode == 'ENABLED'
                                else 'ENABLED')
        _report_error(context, message)
    finally:
        _ROTATE_TOE_BUSY = False


def _update_driver(self, context):
    armature = _active_armature(context) or _fallback_armature(context)
    if self.finger_driver_mode != 'ENABLED':
        if armature is not None:
            finger_limits.remove_finger_drivers(armature)
            finger_limits.apply_finger_limits(armature, self)
        else:
            for obj in context.scene.objects:
                if obj.type == 'ARMATURE':
                    finger_limits.remove_finger_drivers(obj)
                    finger_limits.apply_finger_limits(obj, self)
        return
    if self.finger_limits_mode != 'ENABLED':
        self.finger_driver_mode = 'DISABLED'
        return
    if armature is None:
        self.finger_driver_mode = 'DISABLED'
        return
    finger_limits.apply_finger_drivers(armature, self, context.scene)


def _update_enabled(self, context):
    armature = _active_armature(context)
    if self.finger_limits_mode != 'ENABLED':
        # 先关闭驱动器（同时恢复约束范围），再删除手指约束。
        if self.finger_driver_mode == 'ENABLED':
            self.finger_driver_mode = 'DISABLED'
        if armature is not None:
            finger_limits.remove_finger_drivers(armature)
            finger_limits.remove_finger_limits(armature)
        else:
            for obj in context.scene.objects:
                if obj.type == 'ARMATURE':
                    finger_limits.remove_finger_drivers(obj)
                    finger_limits.remove_finger_limits(obj)
        return
    if armature is None:
        self.finger_limits_mode = 'DISABLED'
        return
    _apply(self, context, armature)


class RGBMOCAP_Properties(bpy.types.PropertyGroup):
    folder: bpy.props.StringProperty(
        name='输出目录',
        description='包含 final.npz 与 toyxyz / RGB MoCap 工程缓存的 FlowHMR 输出目录',
        subtype='DIR_PATH',
    )
    show_finger_limits: bpy.props.BoolProperty(
        name='展开手指骨骼设置',
        description='展开后可在下方修改六组基础手指的 XYZ 最小/最大角度',
        default=False,
    )
    rotate_toe_mode: bpy.props.EnumProperty(
        name='脚尖旋转',
        description='开启时把 つま先 旋转交给 足先EX，关闭时反向交回 つま先',
        items=ROTATE_TOE_MODE_ITEMS,
        default='DISABLED',
        update=_update_rotate_toe,
    )
    finger_limits_mode: bpy.props.EnumProperty(
        name='手指约束',
        description='启用时创建 限定旋转 约束，关闭时删除本插件添加的约束',
        items=FINGER_LIMITS_MODE_ITEMS,
        default='DISABLED',
        update=_update_enabled,
    )
    finger_driver_mode: bpy.props.EnumProperty(
        name='手指约束驱动器',
        description='启用后限制中指与无名指第一节的 Z 旋转，避免交叉；关闭时删除驱动器',
        items=FINGER_DRIVER_MODE_ITEMS,
        default='DISABLED',
        update=_update_driver,
    )

    thumb0_x_min: bpy.props.FloatProperty(name='拇指0 X 最小', default=math.radians(-60.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    thumb0_x_max: bpy.props.FloatProperty(name='拇指0 X 最大', default=math.radians(30.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    thumb0_y_min: bpy.props.FloatProperty(name='拇指0 Y 最小', default=math.radians(-60.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    thumb0_y_max: bpy.props.FloatProperty(name='拇指0 Y 最大', default=math.radians(30.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    thumb0_z_min: bpy.props.FloatProperty(name='拇指0 Z 最小', default=math.radians(-35.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    thumb0_z_max: bpy.props.FloatProperty(name='拇指0 Z 最大', default=math.radians(25.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')

    thumb1_x_min: bpy.props.FloatProperty(name='拇指1 X 最小', default=math.radians(-60.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    thumb1_x_max: bpy.props.FloatProperty(name='拇指1 X 最大', default=math.radians(30.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    thumb1_y_min: bpy.props.FloatProperty(name='拇指1 Y 最小', default=math.radians(-70.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    thumb1_y_max: bpy.props.FloatProperty(name='拇指1 Y 最大', default=math.radians(20.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    thumb1_z_min: bpy.props.FloatProperty(name='拇指1 Z 最小', default=math.radians(-65.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    thumb1_z_max: bpy.props.FloatProperty(name='拇指1 Z 最大', default=math.radians(50.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')

    thumb2_x_min: bpy.props.FloatProperty(name='拇指2 X 最小', default=math.radians(-30.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    thumb2_x_max: bpy.props.FloatProperty(name='拇指2 X 最大', default=math.radians(14.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    thumb2_y_min: bpy.props.FloatProperty(name='拇指2 Y 最小', default=math.radians(0.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    thumb2_y_max: bpy.props.FloatProperty(name='拇指2 Y 最大', default=math.radians(0.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    thumb2_z_min: bpy.props.FloatProperty(name='拇指2 Z 最小', default=math.radians(0.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    thumb2_z_max: bpy.props.FloatProperty(name='拇指2 Z 最大', default=math.radians(0.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')

    index1_x_min: bpy.props.FloatProperty(name='食指1 X 最小', default=math.radians(-90.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    index1_x_max: bpy.props.FloatProperty(name='食指1 X 最大', default=math.radians(30.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    index1_y_min: bpy.props.FloatProperty(name='食指1 Y 最小', default=math.radians(-6.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    index1_y_max: bpy.props.FloatProperty(name='食指1 Y 最大', default=math.radians(12.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    index1_z_min: bpy.props.FloatProperty(name='食指1 Z 最小', default=math.radians(-30.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    index1_z_max: bpy.props.FloatProperty(name='食指1 Z 最大', default=math.radians(15.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')

    index2_x_min: bpy.props.FloatProperty(name='食指2 X 最小', default=math.radians(-140.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    index2_x_max: bpy.props.FloatProperty(name='食指2 X 最大', default=math.radians(25.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    index2_y_min: bpy.props.FloatProperty(name='食指2 Y 最小', default=math.radians(0.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    index2_y_max: bpy.props.FloatProperty(name='食指2 Y 最大', default=math.radians(0.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    index2_z_min: bpy.props.FloatProperty(name='食指2 Z 最小', default=math.radians(0.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    index2_z_max: bpy.props.FloatProperty(name='食指2 Z 最大', default=math.radians(0.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')

    index3_x_min: bpy.props.FloatProperty(name='食指3 X 最小', default=math.radians(-85.0),
                                          min=_LIMIT_MIN, max=_LIMIT_MAX, update=_update_limit, precision=0, subtype='ANGLE')
    index3_x_max: bpy.props.FloatProperty(name='食指3 X 最大', default=math.radians(10.0),
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
