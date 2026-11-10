"""Sidebar panel for the SMPL-H retarget add-on."""
import bpy


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
        col = layout.column()
        row = col.row(align=True)
        split = row.split(factor=0.4, align=True)
        split.alignment = 'RIGHT'
        split.label(text='单侧脚间距')
        right = split.row(align=True)
        right.prop(props, 'foot_spacing', text='')
        right.operator('flowhmr2vmd.adjust_foot_spacing', text='',
                       icon='ARROW_LEFTRIGHT')
        _toggle_row(col, '脚尖旋转', 'flowhmr2vmd.set_toe_rotation',
                    props.toe_rotation == 'ENABLED')
        _toggle_row(col, '手指修复', 'flowhmr2vmd.set_finger_fix',
                    props.finger_fix == 'ENABLED')
        col.operator('flowhmr2vmd.clean_ik_keys', text='清理足部关键帧',
                     icon='ACTION')
        col.operator('flowhmr2vmd.clean_body_keys', text='清理身体关键帧',
                     icon='BONE_DATA')
        row = col.row(align=True)
        split = row.split(factor=0.4, align=True)
        split.alignment = 'RIGHT'
        split.label(text='位移增强')
        right = split.row(align=True)
        right.prop(props, 'boost_axis', text='')
        right.prop(props, 'boost_strength', text='')
        right.operator('flowhmr2vmd.boost_bone_motion', text='',
                       icon='ARROW_LEFTRIGHT')
