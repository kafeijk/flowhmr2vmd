# Vendored RGB MoCap solver

These modules are adapted from RGB MoCap (MIT License):

- `solver.py`
- `contact_trajectory.py`
- `contact_arm_plan.py`
- `foot_calibration.py`

Source: https://github.com/toyxyz/toyxyz_RGBMocap

They run in an external Python 3 process with numpy and scipy. The Blender
add-on does not import them directly because Blender's bundled Python does not
ship scipy.
