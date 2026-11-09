"""Operator orchestration: source -> external solver -> armature keyframes."""
import shutil
import traceback
import bpy
import numpy as np

from ..core import bake, finger_limits, geometry, mmd, smplh, solver_client
from ..core.constants import A, NATIVE_MAPPING, OPTIONAL_MAPPING_INDEX
from .bone_operators import convert_toe_rotation, standardize_channels


def _reset_to_rest(root, armature):
    """把角色恢复到静置状态，保证每次重定向从同一基准开始。

    会清掉上一次的动作、残留的 pose 变换、IK 开关关键帧，以及
    root / 骨架对象的位置旋转；缩放保留（PMX 导入常带缩放）。
    """
    for obj in (root, armature):
        if obj is None:
            continue
        obj.location = (0.0, 0.0, 0.0)
        obj.rotation_mode = 'XYZ'
        obj.rotation_euler = (0.0, 0.0, 0.0)
        obj.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        obj.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
    if armature.animation_data is not None:
        armature.animation_data_clear()
    for pose_bone in armature.pose.bones:
        pose_bone.location = (0.0, 0.0, 0.0)
        pose_bone.rotation_mode = 'QUATERNION'
        pose_bone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pose_bone.rotation_euler = (0.0, 0.0, 0.0)
        pose_bone.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
        pose_bone.scale = (1.0, 1.0, 1.0)
        if hasattr(pose_bone, 'mmd_ik_toggle'):
            pose_bone.mmd_ik_toggle = True
    bpy.context.view_layer.update()


class FLOWHMR2VMD_OT_retarget_smplh(bpy.types.Operator):
    bl_idname = 'flowhmr2vmd.retarget_smplh'
    bl_label = '执行 VMD 重定向'
    bl_description = '读取 final.npz 标记的 toyxyz / RGB MoCap 工程缓存，直接写入选中 MMD 模型骨架的关键帧'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        props = getattr(scene, 'flowhmr2vmd_properties', None)
        folder = bpy.path.abspath(getattr(props, 'folder', '') if props else '')
        if not folder:
            self.report({'ERROR'}, '请选择包含 final.npz 与 toyxyz 工程缓存的输出目录。')
            return {'CANCELLED'}
        from pathlib import Path
        path = Path(folder)
        if not path.is_dir():
            self.report({'ERROR'}, '所选文件夹不存在。')
            return {'CANCELLED'}

        final_path = smplh.find_final_npz(path)
        project = smplh.find_project(path)
        if project is None:
            if final_path is None:
                self.report({'ERROR'},
                            '在所选文件夹及其子文件夹中未找到 final.npz，'
                            '也未找到 corrected_initial_mixamo.json / initial_mixamo.json。')
            else:
                self.report({'ERROR'},
                            f'找到 {final_path.name}，但缺少 toyxyz 工程缓存：需要 '
                            'corrected_initial_mixamo.json、rig.npy、rig-asset.npz、'
                            'surface.npy 和 SMPL-H model.npz。')
            return {'CANCELLED'}
        try:
            source = smplh.prepare_project(project)
        except smplh.SourceError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        if not source.get('raw_file'):
            self.report({'WARNING'},
                        '工程缺少 surface 文件（raw_file），将跳过表面证据阶段，'
                        '脚部效果与 toyxyz 不完全一致。')

        selected = list(context.selected_objects)
        obj = context.active_object if context.active_object in selected else (
            selected[0] if selected else None)
        if obj is None:
            self.report({'ERROR'}, '请先在视图中选中一个 MMD 模型对象。')
            return {'CANCELLED'}
        if not mmd.mmd_tools_enabled():
            self.report({'ERROR'}, '请先启用 MMD Tools 插件。')
            return {'CANCELLED'}
        root, armature = mmd.find_root_and_armature(obj)
        if root is None:
            self.report({'ERROR'}, '选中的对象不属于 MMD 模型，未找到 PMX Root。')
            return {'CANCELLED'}
        if armature is None:
            self.report({'ERROR'}, '在 PMX Root 下未找到骨架。')
            return {'CANCELLED'}

        saved_location = root.location.copy()
        _reset_to_rest(root, armature)
        result = self._run(context, path, project, source, root, armature,
                           props)
        root.location = saved_location
        return result

    def _run(self, context, path, project, source, root, armature, props):
        mapping = list(NATIVE_MAPPING)
        bones = armature.data.bones
        has_upper1 = bones.get('上半身1') is not None
        # 任一侧缺少 つま先 就按两侧都没有处理，只留一半没有意义。
        has_toe = all(bones.get(f'つま先{side}') is not None
                      for side in ('.L', '.R'))
        missing = [name for index, name in enumerate(mapping)
                   if index not in OPTIONAL_MAPPING_INDEX and
                   bones.get(name) is None]
        if not has_toe:
            # 没有 つま先 时，足先EX 是必备的次标准脚尖骨骼。
            missing.extend(f'足先EX{side}' for side in ('.L', '.R')
                           if bones.get(f'足先EX{side}') is None)
        if missing:
            self.report({'ERROR'}, '骨架缺少以下必需骨骼：' + '、'.join(missing))
            return {'CANCELLED'}
        if not mmd.meshes_for(armature):
            self.report({'ERROR'}, '骨架下没有蒙皮网格，无法计算脚底接触。')
            return {'CANCELLED'}
        skip_indices = []
        if not has_upper1:
            # 上半身1 缺失：丢弃该槽位，Spine1 的旋转已包含在 上半身2
            # 的绝对世界旋转里，效果等同并入 上半身2。
            mapping[6] = '上半身2'
            skip_indices.append(6)
        if not has_toe:
            skip_indices.extend((10, 11))
        meshes = mmd.meshes_for(armature)
        rest = np.empty((22, 3), np.float64)

        def head(name):
            return np.array(armature.matrix_world @
                            armature.data.bones[name].head_local) @ A

        for index, name in enumerate(mapping):
            if index == 6 and not has_upper1:
                rest[index] = (head('上半身') + head('上半身2')) * .5
            elif index in (10, 11) and not has_toe:
                rest[index] = head('足先EX' + ('.L' if index == 10 else '.R'))
            else:
                rest[index] = head(name)

        try:
            contact = geometry.build_contact_geometry(armature, meshes, mapping, rest)
        except geometry.GeometryError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        request_project = source['project']
        temp = solver_client.make_temp()
        try:
            request = temp / 'request.npz'
            solver_client.write_request(
                request, source, rest, mapping, contact,
                source['source_names'], False, project=request_project)
            stance_path, score = solver_client.run_solver(request, temp, path)
            stance = solver_client.read_stance(stance_path)
            result = bake.bake_solution(
                armature, mapping, stance, source, skip_indices=skip_indices)
        except solver_client.SolverError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        except Exception as exc:
            detail = traceback.format_exc()
            print(detail)
            self.report({'ERROR'}, f'生成动作失败：{exc}')
            return {'CANCELLED'}
        finally:
            shutil.rmtree(temp, ignore_errors=True)

        # 重定向后自动把 腰 旋转与 全ての親 位移并入 センター。
        standardize_channels(armature)
        if (props is not None and props.rotate_toe_mode == 'ENABLED' and
                has_toe):
            convert_toe_rotation(armature, True)

        if props is not None and props.finger_limits_mode == 'ENABLED':
            finger_limits.apply_finger_limits(armature, props)
            if props.finger_driver_mode == 'ENABLED':
                finger_limits.apply_finger_drivers(armature, props, context.scene)

        text = (f"重定向成功，共{result['frames']} 帧。")
        if not has_upper1:
            text += ' 缺少 上半身1：其旋转已并入 上半身2。'
        if not has_toe:
            text += ' 缺少 つま先（任一侧缺失按两侧处理）：两侧脚趾旋转已丢弃。'
        self.report({'INFO'}, text)
        if score.get('pt_fallback'):
            self.report({'WARNING'},
                        '读取 corrected_initial.pt 失败，已用工程 rig 旋转代替 native 旋转：'
                        + str(score['pt_fallback'])[-200:])
        if result['skipped_toe']:
            self.report({'WARNING'},
                        '足先EX 位于付与链下，脚趾未单独打键，会跟随脚踝刚性运动。')
        return {'FINISHED'}


class FLOWHMR2VMD_OT_reset_finger_limits(bpy.types.Operator):
    bl_idname = 'flowhmr2vmd.reset_finger_limits'
    bl_label = '恢复默认手指约束'
    bl_description = '把所有手指角度范围恢复为默认值'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = getattr(context.scene, 'flowhmr2vmd_properties', None)
        if props is None:
            self.report({'ERROR'}, '找不到插件属性。')
            return {'CANCELLED'}
        rna = props.bl_rna.properties
        for name in finger_limits.property_names():
            item = rna.get(name)
            if item is not None:
                setattr(props, name, item.default)
        self.report({'INFO'}, '手指约束的角度范围已恢复为默认值。')
        return {'FINISHED'}
