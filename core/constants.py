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

# Fallback for standard MMD models without 上半身1 / つま先.
# Index 6 repeats 上半身; the baker keeps the first occurrence and treats
# index 6 as a virtual middle-spine point.
FALLBACK_MAPPING = list(NATIVE_MAPPING)
FALLBACK_MAPPING[6] = '上半身'
FALLBACK_MAPPING[10] = '足先EX.L'
FALLBACK_MAPPING[11] = '足先EX.R'

MAPPING_ITEMS = [
    ('NATIVE', 'RGB MoCap 原生（上半身1 / つま先）',
     '严格使用 RGB MoCap 原始骨骼名；模型必须同时具备 上半身1 和 つま先'),
    ('FALLBACK', '兼容写法（上半身 / 足先EX）',
     '用 上半身 代替 上半身1，用 足先EX 代替 つま先；适用于大多数标准 MMD 模型'),
]


def mapping_for(kind):
    return list(FALLBACK_MAPPING if kind == 'FALLBACK' else NATIVE_MAPPING)
