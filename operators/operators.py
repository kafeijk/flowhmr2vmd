"""Operator orchestration: source -> external solver -> armature keyframes."""
import shutil
import traceback
import bpy
import numpy as np

from ..core import bake, finger_limits, geometry, mmd, smplh, solver_client
from ..core.constants import A, MAPPING_ITEMS, mapping_for


class RGBMOCAP_OT_retarget_smplh(bpy.types.Operator):
    bl_idname = 'rgbmocap.retarget_smplh'
    bl_label = '执行 VMD 重定向'
    bl_description = '读取 final.npz 标记的 toyxyz / RGB MoCap 工程缓存，直接写入选中 MMD 模型骨架的关键帧'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        props = getattr(scene, 'rgbmocap_properties', None)
        folder = bpy.path.abspath(getattr(props, 'folder', '') if props else '')
        mapping_kind = getattr(props, 'mapping', 'FALLBACK') if props else 'FALLBACK'
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

        mapping = mapping_for(mapping_kind)
        message = mmd.validate_mapping(armature, mapping)
        if message:
            # 选错模式时自动回退到能用的那一套，避免模型缺 上半身1/つま先
            # 时直接失败；能支持的模型优先用原生映射。
            other = 'FALLBACK' if mapping_kind == 'NATIVE' else 'NATIVE'
            alternative = mapping_for(other)
            if mmd.validate_mapping(armature, alternative) is None:
                mapping_kind = other
                mapping = alternative
                self.report({'INFO'},
                            f'当前模型不支持所选映射，已自动改用 {other}：{message}')
            else:
                label = dict((key, name) for key, name, _ in MAPPING_ITEMS)[
                    mapping_kind]
                self.report({'ERROR'}, f'{message}（当前对照：{label}）')
                return {'CANCELLED'}

        meshes = mmd.meshes_for(armature)
        rest = np.empty((22, 3), np.float64)

        def head(name):
            return np.array(armature.matrix_world @
                            armature.data.bones[name].head_local) @ A

        for index, name in enumerate(mapping):
            if index == 6 and mapping_kind == 'FALLBACK':
                rest[index] = (head('上半身') + head('上半身2')) * .5
            else:
                rest[index] = head(name)

        lock_toes = (mapping[10] == '足先EX.L' and
                     mmd.toe_is_granted(armature, mapping))
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
                source['source_names'], lock_toes, project=request_project)
            stance_path, score = solver_client.run_solver(request, temp, path)
            stance = solver_client.read_stance(stance_path)
            result = bake.bake_solution(
                armature, mapping, stance, source,
                skip_toe=(mapping[10] == '足先EX.L' and lock_toes))
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

        if props is not None and props.finger_limits_mode == 'ENABLED':
            finger_limits.apply_finger_limits(armature, props)
            if props.finger_driver_mode == 'ENABLED':
                finger_limits.apply_finger_drivers(armature, props, context.scene)

        contacts = score.get('score', {}).get('contacts') or []
        details = []
        for label, row in (('左脚', contacts[0] if len(contacts) > 0 else None),
                           ('右脚', contacts[1] if len(contacts) > 1 else None)):
            if row:
                details.append(f"{label}接触平均 {row.get('position_mean_mm', 0):.3f} mm / "
                               f"p95 {row.get('position_p95_mm', 0):.3f} mm")
        source_text = 'RGB MoCap 工程 ' + project['kind']
        text = (f"来源：{source_text}。动作已写入骨架：{result['action']}，"
                f"{result['frames']} 帧。")
        if details:
            text += ' ' + '；'.join(details) + '。'
        self.report({'INFO'}, text)
        if score.get('pt_fallback'):
            self.report({'WARNING'},
                        '读取 corrected_initial.pt 失败，已用工程 rig 旋转代替 native 旋转：'
                        + str(score['pt_fallback'])[-200:])
        if result['skipped_toe']:
            self.report({'WARNING'},
                        '足先EX 位于付与链下，脚趾未单独打键，会跟随脚踝刚性运动。')
        if result['muted']:
            self.report({'WARNING'},
                        f"已临时关闭 {len(set(result['muted']))} 根骨骼的 IK/付与约束，"
                        '以便烘焙的 FK 动作在视口中生效。')
        return {'FINISHED'}


class RGBMOCAP_OT_reset_finger_limits(bpy.types.Operator):
    bl_idname = 'rgbmocap.reset_finger_limits'
    bl_label = '恢复默认手指约束'
    bl_description = '把所有手指角度范围恢复为默认值'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = getattr(context.scene, 'rgbmocap_properties', None)
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
