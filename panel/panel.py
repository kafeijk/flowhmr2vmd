"""Sidebar panel for the SMPL-H retarget add-on."""
import bpy

from ..core import mmd
from ..core.finger_limits import FINGER_GROUPS
from ..operators.bone_operators import get_bone_progress


def _model_state(context):
    """当前选中 MMD 模型根对象上记录的开关状态，未选中时全部关闭。"""
    selected = list(context.selected_objects)
    obj = context.active_object if context.active_object in selected else (
        selected[0] if selected else None)
    if obj is None:
        return mmd.read_state(None)
    root, armature = mmd.find_root_and_armature(obj)
    return mmd.read_state(root if armature is not None else None)


def _toggle_row(layout, label, op_idname, enabled):
    """左侧固定宽度标签，右侧两个等宽按钮，保证多行对齐。"""
    row = layout.row(align=True)
    split = row.split(factor=0.4, align=True)
    split.alignment = 'RIGHT'
    split.label(text=label)
    buttons = split.row(align=True)
    halves = buttons.split(factor=0.5, align=True)
    halves.operator(op_idname, text='开启',
                    depress=enabled).mode = 'ENABLED'
    halves.operator(op_idname, text='关闭',
                    depress=not enabled).mode = 'DISABLED'
    return row


class FLOWHMR2VMD_PT_panel(bpy.types.Panel):
    bl_label = 'VMD 动作重定向'
    bl_idname = 'FLOWHMR2VMD_PT_retarget'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'FlowHMR2VMD'
    bl_order = 0

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        scene = context.scene
        props = scene.flowhmr2vmd_properties
        col = layout.column()
        col.prop(props, 'folder', text='动作目录')


        row = col.row(align=True)
        row.operator('flowhmr2vmd.retarget_smplh', text='执行', icon='PLAY')

class FLOWHMR2VMD_PT_tools(bpy.types.Panel):
    bl_label = '小功能'
    bl_idname = 'FLOWHMR2VMD_PT_tools'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'FlowHMR2VMD'
    bl_order = 1

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        scene = context.scene
        props = scene.flowhmr2vmd_properties
        state = _model_state(context)
        col = layout.column()
        _toggle_row(col, '脚尖旋转', 'flowhmr2vmd.set_toe_rotation',
                    props.toe_rotation == 'ENABLED')
        progress = get_bone_progress()
        if progress['active']:
            if hasattr(col, 'progress'):
                col.progress(factor=progress['factor'], type='BAR',
                             text=progress['text'])
            else:
                col.label(text=progress['text'])

        row = col.row(align=True)
        row.prop(props, 'show_finger_limits', text='展开手指骨骼设置')
        if props.show_finger_limits:
            refresh = row.row(align=True)
            refresh.operator('flowhmr2vmd.reset_finger_limits', text='', icon='FILE_REFRESH')

        if props.show_finger_limits:
            box = col.box()
            inner = box.column()
            inner.use_property_decorate = False

            _toggle_row(inner, '手指约束', 'flowhmr2vmd.finger_limits',
                        state[mmd.STATE_FINGER_LIMITS])
            row = _toggle_row(inner, '手指约束驱动器',
                              'flowhmr2vmd.finger_driver',
                              state[mmd.STATE_FINGER_DRIVER])
            row.enabled = state[mmd.STATE_FINGER_LIMITS]
            row = inner.row(align=True)
            row.operator('flowhmr2vmd.bake_fingers', text='烘焙手指动作',
                         icon='ACTION')
            inner.separator()

            for label, prefix in FINGER_GROUPS:
                for axis in ('x', 'y', 'z'):
                    row = inner.row(align=True)
                    row.prop(props, f'{prefix}_{axis}_min',
                             text=f'{label} {axis.upper()}' if axis == 'x' else axis.upper())
                    tilde = row.row(align=True)
                    tilde.use_property_split = False
                    tilde.ui_units_x = 0.6
                    tilde.alignment = 'CENTER'
                    tilde.label(text='~')
                    row.prop(props, f'{prefix}_{axis}_max', text='')
                inner.separator()
            hint_row = inner.row()
            hint_split = hint_row.split(factor=0.4)
            hint_split.label(text='')
            hint = hint_split.column(align=True)
            hint.label(text='其余手指沿用食指范围（暂定）', icon='INFO')
            hint.label(text='右侧手指参考左侧', icon='INFO')
