"""Shared constants and bone-name mappings.

Portions adapted from RGB MoCap (https://github.com/toyxyz/toyxyz_RGBMocap),
MIT License. Copyright (c) 2026 RGB MoCap contributors.
"""
import numpy as np

# Row-vector transform from Blender Z-up coordinates to the Y-up solver frame.
A = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

# Source skeleton order used by RGB MoCap (Mixamo-style labels).
NAMES = ['Hips', 'LeftUpLeg', 'RightUpLeg', 'Spine', 'LeftLeg', 'RightLeg',
         'Spine1', 'LeftFoot', 'RightFoot', 'Spine2', 'LeftToeBase', 'RightToeBase',
         'Neck', 'LeftShoulder', 'RightShoulder', 'Head', 'LeftArm', 'RightArm',
         'LeftForeArm', 'RightForeArm', 'LeftHand', 'RightHand']

PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
                    16, 17, 18, 19])

# The exact list used by RGB MoCap's retarget/mapping.py.
NATIVE_MAPPING = ['下半身', '足.L', '足.R', '上半身', 'ひざ.L', 'ひざ.R', '上半身1',
                  '足首.L', '足首.R', '上半身2', 'つま先.L', 'つま先.R', '首',
                  '肩.L', '肩.R', '頭', '腕.L', '腕.R', 'ひじ.L', 'ひじ.R',
                  '手首.L', '手首.R']

# 可选骨骼：缺失时可以丢弃，其余骨骼必须存在。
OPTIONAL_MAPPING_INDEX = (6, 10, 11)
