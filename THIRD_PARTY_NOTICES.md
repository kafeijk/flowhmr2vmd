# Third-party notices

## RGB MoCap

This add-on adapts algorithms, constants and code structure from RGB MoCap:

- https://github.com/toyxyz/toyxyz_RGBMocap
- MIT License, Copyright (c) 2026 RGB MoCap contributors.

Adapted parts include the SMPL-H retarget baseline, contact target planning,
foot calibration, body/finger baking and VMD-oriented bone handling. The
vendored solver files under `vendor/mocap/` are copies of RGB MoCap modules
(MIT) and retain their original notices.

## MMD Tools

The add-on requires Blender's MMD Tools add-on:

- https://github.com/UuuNyaa/blender_mmd_tools
- GPL-3.0-or-later.

MMD Tools is not bundled with this add-on. It must be installed and enabled
separately in Blender.

## SMPL-H body model

The add-on reads a user-provided SMPL-H `model.npz`. The model is not bundled
and remains subject to its own license and registration terms.
