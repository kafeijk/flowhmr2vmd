"""FlowHMR2VMD.

Reads a FlowHMR final.npz or an RGB MoCap project cache, retargets the
SMPL-H motion to the selected MMD armature, and writes keyframes directly on
that armature.

MIT License. Portions adapted from RGB MoCap:
https://github.com/toyxyz/toyxyz_RGBMocap
Copyright (c) 2026 来杯咖啡再说.
"""
import bpy

from ...common.class_loader import auto_load
from ...common.class_loader.auto_load import add_properties, remove_properties

bl_info = {
    'name': 'FlowHMR2VMD',
    'author': '来杯咖啡再说',
    'version': (1, 0, 0),
    'blender': (4, 2, 0),
    'location': 'View3D > Sidebar > FlowHMR2VMD',
    'description': '读取 final.npz 或 RGB MoCap 工程缓存，把 SMPL-H 动作重定向到选中的 MMD 模型骨架并直接打关键帧',
    'category': 'Animation',
}

_addon_properties = {}


def register():
    auto_load.init()
    auto_load.register()
    add_properties(_addon_properties)


def unregister():
    auto_load.unregister()
    remove_properties(_addon_properties)


if __name__ == '__main__':
    register()
