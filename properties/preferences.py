"""User preferences: external Python environment path."""
import bpy


def addon_module():
    """当前插件的 Blender 模块名（偏好设置按模块名挂载）。"""
    parts = (__package__ or __name__).split('.')
    if parts[0] == 'bl_ext' and len(parts) >= 3:
        return '.'.join(parts[:3])
    return parts[0]


def get_env_path():
    """偏好设置里填写的环境地址，未填写时返回空字符串。"""
    try:
        addon = bpy.context.preferences.addons.get(addon_module())
        preferences = getattr(addon, 'preferences', None)
        return (getattr(preferences, 'env_path', '') or '').strip()
    except Exception:
        return ''


class FLOWHMR2VMD_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = addon_module()

    env_path: bpy.props.StringProperty(
        name='环境地址',
        description='优先在该目录及其子目录中查找带 scipy / torch 的 Python环境',
        subtype='DIR_PATH',
        default='',
    )

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        col = layout.column()
        col.prop(self, 'env_path')