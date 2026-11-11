"""Operator orchestration: source -> external solver -> armature keyframes."""
import shutil
import traceback
from pathlib import Path

import bpy
import numpy as np

from .bone_operators import (apply_finger_fix, clear_toe_bone_curves,
                             convert_toe_rotation, restore_toe_from_custom,
                             standardize_channels, store_toe_to_custom)
from ..core import bake, geometry, mmd, smplh, solver_client
from ..core.constants import A, NATIVE_MAPPING, OPTIONAL_MAPPING_INDEX


def _middle_spine_name(bones):
    """
    上半身 与 上半身2 之间唯一的中间骨（上半身1 / 上半身3 等）。
    两者之间没有骨骼或不止一个时返回 None。
    """
    lower = bones.get('上半身')
    upper = bones.get('上半身2')
    if lower is None or upper is None:
        return None
    chain = []
    parent = upper.parent
    while parent is not None and parent.name != lower.name:
        chain.append(parent.name)
        parent = parent.parent
    if parent is None or len(chain) != 1:
        return None
    return chain[0]


def _reset_to_rest(root, armature):
    """把角色恢复到静置状态，保证每次重定向从同一基准开始。"""
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
    bl_label = '执行'
    bl_description = '将FlowHMR生成的动作数据转换为VMD格式，并将关键帧注册到当前选中模型的骨骼上'
    bl_options = {'REGISTER', 'UNDO'}

    def check_props(self, props):
        """在执行重定向之前的校验"""
        if not mmd.mmd_tools_enabled():
            self.report({'ERROR'}, '请启用 MMD Tools 插件！')
            return False, None

        roots = set()
        for obj in bpy.context.selected_objects:
            root = mmd.find_pmx_root_with_child(obj)
            if root is not None:
                roots.add(root)
        if not roots:
            self.report({'ERROR'}, '请选择 MMD 模型！')
            return False, None
        if len(roots) > 1:
            self.report({'ERROR'}, '请选择单个 MMD 模型！')
            return False, None
        root = roots.pop()
        armature = mmd.find_armature(root)
        if armature is None:
            self.report({'ERROR'}, '未找到模型骨架！')
            return False, None

        folder = bpy.path.abspath(props.folder)
        if not folder:
            self.report({'ERROR'}, '请选择 FlowHMR 动作目录！')
            return False, None
        path = Path(folder)
        if not path.is_dir():
            self.report({'ERROR'}, '动作目录不存在！')
            return False, None

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
            return False, None
        try:
            source = smplh.prepare_project(project)
        except smplh.SourceError as exc:
            self.report({'ERROR'}, str(exc))
            return False, None
        if not source.get('raw_file'):
            self.report({'WARNING'},
                        '工程缺少 surface 文件（raw_file），将跳过表面证据阶段，'
                        '脚部效果与 toyxyz 不完全一致。')

        bones = armature.data.bones
        missing = [name for index, name in enumerate(NATIVE_MAPPING)
                   if index not in OPTIONAL_MAPPING_INDEX and
                   bones.get(name) is None]
        if missing:
            self.report({'ERROR'}, '骨架缺少以下必需骨骼：' + '、'.join(missing))
            return False, None
        if not mmd.find_pmx_objects(armature):
            self.report({'ERROR'}, '模型网格对象不存在！无法计算脚底接触。')
            return False, None
        return True, dict(root=root, armature=armature, path=path, source=source)

    def execute(self, context):
        props = context.scene.flowhmr2vmd_properties
        ok, data = self.check_props(props)
        if not ok:
            return {'CANCELLED'}
        return self._run(data, props)

    def _run(self, data, props):
        root = data['root']
        armature = data['armature']
        path = data['path']
        source = data['source']

        bones = armature.data.bones
        # 脊柱中间骨命名不唯一，动态获取
        middle_spine = _middle_spine_name(bones)
        has_upper1 = middle_spine is not None
        # 任一侧缺少 つま先、足先EX 就按两侧都没有处理，只留一半没有意义。
        has_toe = all(bones.get(f'つま先{side}') is not None for side in ('.L', '.R'))
        has_ex = all(bones.get(f'足先EX{side}') is not None for side in ('.L', '.R'))
        # 基于 RGB MoCap 项目的参与重定向的骨骼
        mapping = list(NATIVE_MAPPING)
        skip_indices = []
        if has_upper1:
            # 替换脊柱中间骨名称
            mapping[6] = middle_spine
        else:
            # 中间骨不存在则丢弃，Spine1 的旋转已包含在 上半身2 的绝对世界旋转里，效果等同并入 上半身2
            mapping[6] = '上半身2'
            skip_indices.append(6)
        if not has_toe:
            # 中当模型不存在つま先但存在足先EX时，将脚尖关键帧写入足先EX的自定义属性中
            if has_ex:
                mapping[10] = '足先EX.L'
                mapping[11] = '足先EX.R'
            else:
                # 若足先EX也不存在，则丢弃脚尖运动
                skip_indices.extend((10, 11))

        saved_location = root.location.copy()
        _reset_to_rest(root, armature)
        objects = mmd.find_pmx_objects(armature)
        rest = np.empty((22, 3), np.float64)

        def head(name):
            return np.array(armature.matrix_world @
                            armature.data.bones[name].head_local) @ A

        for index, name in enumerate(mapping):
            if index == 6 and not has_upper1:
                rest[index] = (head('上半身') + head('上半身2')) * .5
            elif index in (10, 11) and not has_toe:
                side = '.L' if index == 10 else '.R'
                rest[index] = head(('足先EX' if has_ex else '足首') + side)
            else:
                rest[index] = head(name)

        try:
            contact = geometry.build_contact_geometry(armature, objects, mapping, rest)
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

        # 重定向后执行标准化
        standardize_channels(armature)

        # 若脚尖旋转等可选项被勾选，则重定向后默认执行一次
        toe_enabled = props.toe_rotation == 'ENABLED'
        if not has_toe and has_ex:
            store_toe_to_custom(armature)
            if toe_enabled:
                restore_toe_from_custom(armature)
        elif has_toe and toe_enabled:
            convert_toe_rotation(armature, True)
        elif has_toe:
            clear_toe_bone_curves(armature)
        if props.finger_fix == 'ENABLED':
            apply_finger_fix(armature)

        text = f"已完成重定向，共 {result['frames']} 帧。"
        if not has_upper1:
            text += '缺少脊柱中间骨，结果已并入「上半身2」。'
        if not has_toe:
            if has_ex:
                text += '缺少「つま先」，脚尖旋转已存入「足先EX」的自定义通道。'
            else:
                text += '缺少「つま先」和「足先EX」，脚尖数据已丢弃。'
        self.report({'INFO'}, text)

        if score.get('pt_fallback'):
            self.report({'WARNING'},
                        '读取 corrected_initial.pt 失败，已用工程 rig 旋转代替 native 旋转：'
                        + str(score['pt_fallback'])[-200:])

        root.location = saved_location
        return {'FINISHED'}
