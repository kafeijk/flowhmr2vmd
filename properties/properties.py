"""Panel properties and finger-limit constraint state."""
import math

import bpy

from ..core import finger_limits, mmd

_LIMIT_MIN = math.radians(-180.0)
_LIMIT_MAX = math.radians(180.0)


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
    armature = _active_armature(context) or _fallback_armature(context)
    if armature is None:
        return
    root, _ = mmd.find_root_and_armature(armature)
    if root is None or not root.get(mmd.STATE_FINGER_LIMITS, False):
        return
    _apply(self, context, armature)


_TOE_ROTATION_ITEMS = (
    ('ENABLED', '开启', '开启后把脚尖旋转写入 足先EX 骨骼'),
    ('DISABLED', '关闭', '关闭后清空 足先EX 骨骼关键帧'),
)
_updating_toe = False


def _show_toe_error(context, message):
    if bpy.app.background:
        print('脚尖旋转:', message)
        return
    window = getattr(context, 'window', None)
    if window is None:
        return

    def draw(menu, _context):
        menu.layout.label(text=message)

    context.window_manager.popup_menu(draw, title='脚尖旋转', icon='ERROR')


def _update_toe_rotation(self, context):
    global _updating_toe
    if _updating_toe:
        return
    armature = _active_armature(context)
    if armature is None:
        return
    from ..operators.bone_operators import convert_toe_rotation
    enabled = self.toe_rotation == 'ENABLED'
    ok, message = convert_toe_rotation(armature, enabled)
    if ok:
        return
    _updating_toe = True
    try:
        self.toe_rotation = 'DISABLED'
    finally:
        _updating_toe = False
    _show_toe_error(context, message)


class FLOWHMR2VMD_Properties(bpy.types.PropertyGroup):
    folder: bpy.props.StringProperty(
        name='动作目录',
        description='FlowHMR输出的动作所在目录',
        subtype='DIR_PATH',
    )
    toe_rotation: bpy.props.EnumProperty(
        name='脚尖旋转',
        description='开启后把脚尖旋转写入 足先EX；关闭时清空 足先EX 骨骼关键帧',
        items=_TOE_ROTATION_ITEMS,
        default='DISABLED',
        update=_update_toe_rotation,
    )
    show_finger_limits: bpy.props.BoolProperty(
        name='展开手指骨骼设置',
        description='展开手指骨骼设置',
        default=False,
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
        if not hasattr(bpy.types.Scene, 'flowhmr2vmd_properties'):
            bpy.types.Scene.flowhmr2vmd_properties = bpy.props.PointerProperty(
                type=FLOWHMR2VMD_Properties)

    @staticmethod
    def unregister():
        if hasattr(bpy.types.Scene, 'flowhmr2vmd_properties'):
            del bpy.types.Scene.flowhmr2vmd_properties


def register():
    FLOWHMR2VMD_Properties.register()


def unregister():
    FLOWHMR2VMD_Properties.unregister()
