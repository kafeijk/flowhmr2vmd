"""Sidebar panel for the SMPL-H retarget add-on."""
import bpy

from ..core.finger_limits import FINGER_GROUPS


class RGBMOCAP_PT_panel(bpy.types.Panel):
    bl_label = 'VMD 动作重定向'
    bl_idname = 'RGBMOCAP_PT_retarget'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'VMD Retarget'
    bl_order = 0

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        scene = context.scene
        props = scene.rgbmocap_properties
        col = layout.column()
        col.prop(props, 'folder', text='文件夹')
        col.prop(props, 'mapping', text='名称对照')
        col.separator()
        row = col.row(align=True)
        row.operator('rgbmocap.retarget_smplh', text='执行', icon='PLAY')
        col.label(text='先选中 MMD 模型对象；结果为骨架关键帧。', icon='INFO')

        col.separator()
        col.prop(props, 'show_finger_limits', text='展开手指骨骼约束')
        if not props.show_finger_limits and props.finger_limits_enabled:
            col.label(text='手指限制约束：已启用', icon='CHECKMARK')
        if props.show_finger_limits:
            box = layout.box()
            inner = box.column()
            inner.use_property_split = False
            inner.use_property_decorate = False
            inner.label(text='基础范围（.L）')
            for label, prefix in FINGER_GROUPS:
                inner.label(text=label)
                for axis in ('x', 'y', 'z'):
                    row = inner.row(align=True)
                    row.prop(props, f'{prefix}_{axis}_min', text=f'{axis.upper()} 最小')
                    row.prop(props, f'{prefix}_{axis}_max', text='最大')
                inner.separator()
            inner.label(text='中指/薬指/小指沿用对应人指范围；.R 的 X/Y 取反对调，Z 不变。',
                        icon='INFO')
            inner.separator()
            inner.prop(props, 'finger_limits_enabled',
                       text='创建手指限制旋转约束', toggle=True)
            if props.finger_limit_status:
                inner.label(text=props.finger_limit_status, icon='INFO')
