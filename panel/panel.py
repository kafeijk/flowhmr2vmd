"""Sidebar panel for the SMPL-H retarget add-on."""
import bpy

from addons.flowhmr2vmd.operators.bone_operators import (
    FLOWHMR2VMD_OT_adjust_foot_spacing,
    FLOWHMR2VMD_OT_boost_bone_motion,
    FLOWHMR2VMD_OT_clean_body_keys,
    FLOWHMR2VMD_OT_clean_ik_keys,
    FLOWHMR2VMD_OT_set_finger_fix,
    FLOWHMR2VMD_OT_set_toe_rotation,
)
from addons.flowhmr2vmd.operators.operators import FLOWHMR2VMD_OT_retarget_smplh


def _split_row(layout, label):
    """左侧固定宽度标签，返回右侧控件容器。"""
    row = layout.row(align=True)
    split = row.split(factor=0.4, align=True)
    split.alignment = 'RIGHT'
    split.label(text=label)
    return split.row(align=True)


def _toggle_row(layout, operator, enabled):
    """左侧固定宽度标签，右侧两个等宽按钮，保证多行对齐。"""
    buttons = _split_row(layout, operator.bl_label)
    halves = buttons.split(factor=0.5, align=True)
    halves.operator(operator.bl_idname, text='开启',
                    depress=enabled).mode = 'ENABLED'
    halves.operator(operator.bl_idname, text='关闭',
                    depress=not enabled).mode = 'DISABLED'
    return buttons


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
        col.prop(props, 'folder')

        row = col.row(align=True)
        row.operator(FLOWHMR2VMD_OT_retarget_smplh.bl_idname, text=FLOWHMR2VMD_OT_retarget_smplh.bl_label, icon='PLAY')


class FLOWHMR2VMD_PT_postprocess(bpy.types.Panel):
    bl_label = '后处理'
    bl_idname = 'FLOWHMR2VMD_PT_postprocess'
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

        col = layout.column()
        right = _split_row(col, '单侧脚间距')
        right.prop(props, 'foot_spacing', text='')
        right.operator(FLOWHMR2VMD_OT_adjust_foot_spacing.bl_idname, text='', icon='ARROW_LEFTRIGHT')

        _toggle_row(col, FLOWHMR2VMD_OT_set_toe_rotation, props.toe_rotation == 'ENABLED')
        _toggle_row(col, FLOWHMR2VMD_OT_set_finger_fix, props.finger_fix == 'ENABLED')

        col.operator(FLOWHMR2VMD_OT_clean_ik_keys.bl_idname, text=FLOWHMR2VMD_OT_clean_ik_keys.bl_label, icon='ACTION')
        col.operator(FLOWHMR2VMD_OT_clean_body_keys.bl_idname, text=FLOWHMR2VMD_OT_clean_body_keys.bl_label,
                     icon='BONE_DATA')


class FLOWHMR2VMD_PT_tools(bpy.types.Panel):
    bl_label = '小功能'
    bl_idname = 'FLOWHMR2VMD_PT_tools'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'FlowHMR2VMD'
    bl_order = 2

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        props = context.scene.flowhmr2vmd_properties

        col = layout.column()
        right = _split_row(col, '位移增强')
        right.prop(props, 'boost_axis', text='')
        right.prop(props, 'boost_strength', text='')
        right.operator(FLOWHMR2VMD_OT_boost_bone_motion.bl_idname, text='', icon='MOUSE_LMB')
