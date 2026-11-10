"""Panel properties and toe rotation state."""
import bpy

from ..core import mmd


def _active_armature(context):
    obj = context.active_object
    if obj is None or obj not in context.selected_objects:
        selected = list(context.selected_objects)
        obj = selected[0] if selected else obj
    if obj is None:
        return None
    root, armature = mmd.find_root_and_armature(obj)
    return armature if root is not None else None


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


_FINGER_FIX_ITEMS = (
    ('ENABLED', '开启', '开启后清除手指末端多余旋转并补偿第一指节'),
    ('DISABLED', '关闭', '关闭后把手指旋转恢复到修复前'),
)


def _update_finger_fix(self, context):
    armature = _active_armature(context)
    if armature is None:
        return
    from ..operators.bone_operators import (apply_finger_fix,
                                            revert_finger_fix)
    if self.finger_fix == 'ENABLED':
        apply_finger_fix(armature)
    else:
        revert_finger_fix(armature)


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
    foot_spacing: bpy.props.FloatProperty(
        name='脚间距',
        description='调整足ＩＫ时左右各偏移的米数（左向 +X，右向 -X）',
        default=0.0,
        min=-1.0,
        max=1.0,
        step=10,
        precision=2,
    )
    finger_fix: bpy.props.EnumProperty(
        name='手指修复',
        description='开启后清除手指末端多余旋转并补偿第一指节；关闭时恢复修复前',
        items=_FINGER_FIX_ITEMS,
        default='DISABLED',
        update=_update_finger_fix,
    )

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
