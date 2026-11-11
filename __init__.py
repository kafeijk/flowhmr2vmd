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
    'blender': (3, 2, 0),
    'location': 'View3D > Sidebar > FlowHMR2VMD',
    'description': '将FlowHMR生成的动作数据转换为VMD格式，并将关键帧注册到当前选中模型的骨骼上',
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
