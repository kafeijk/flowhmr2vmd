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
        col.prop(props, 'folder', text='NPZ 文件夹')
        col.prop(props, 'mapping', text='骨骼名称映射')
        row = col.row(align=True)
        row.prop(props, 'show_finger_limits', text='展开手指骨骼设置')
        if props.show_finger_limits:
            refresh = row.row(align=True)
            refresh.operator('rgbmocap.reset_finger_limits', text='', icon='FILE_REFRESH')
        if props.show_finger_limits:
            box = col.box()
            inner = box.column()
            inner.use_property_decorate = False

            row = inner.row(align=True)
            row.prop(props, 'finger_limits_mode', expand=True)
            row = inner.row(align=True)
            row.prop(props, 'finger_driver_mode', expand=True)
            row.enabled = props.finger_limits_mode == 'ENABLED'
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

        row = col.row(align=True)
        row.operator('rgbmocap.retarget_smplh', text='执行', icon='PLAY')
