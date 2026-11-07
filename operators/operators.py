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
    bl_description = '从 final.npz 或 RGB MoCap 工程缓存生成动作，直接写入选中 MMD 模型骨架的关键帧'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        props = getattr(scene, 'rgbmocap_properties', None)
        folder = bpy.path.abspath(getattr(props, 'folder', '') if props else '')
        mapping_kind = getattr(props, 'mapping', 'FALLBACK') if props else 'FALLBACK'
        if not folder:
            self.report({'ERROR'}, '请选择包含 final.npz 的文件夹。')
            return {'CANCELLED'}
        from pathlib import Path
        path = Path(folder)
        if not path.is_dir():
            self.report({'ERROR'}, '所选文件夹不存在。')
            return {'CANCELLED'}

        final_path = smplh.find_final_npz(path)
        project = None
        final = None
        fkdata = None
        if final_path is not None:
            try:
                final = smplh.load_final(final_path)
            except smplh.SourceError as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}
            model_path = smplh.find_smplh_model(path, final['gender'])
            if model_path is None:
                self.report({'ERROR'},
                            '未找到 SMPL-H 模型 model.npz。请把 FlowHMR 工具包的 '
                            'app/flowhmr/assets/body_models/smplh/neutral/model.npz '
                            '保留在输出目录的上级，或设置环境变量 RGBMOCAP_SMPLH_MODEL。')
                return {'CANCELLED'}
            try:
                model = smplh.load_model(model_path)
                fkdata = smplh.forward_kinematics(final, model)
            except smplh.SourceError as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}
        else:
            project = smplh.find_project(path)
            if project is None:
                self.report({'ERROR'},
                            '在所选文件夹及其子文件夹中未找到 final.npz，'
                            '也未找到 corrected_initial_mixamo.json / initial_mixamo.json。')
                return {'CANCELLED'}

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

        if project is None:
            source = smplh.prepare_source(final, fkdata, lock_toes=lock_toes)
            request_project = None
        else:
            try:
                source = smplh.prepare_project(project)
            except smplh.SourceError as exc:
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
            count, missing = finger_limits.apply_finger_limits(armature, props)
            status = f'已创建 {count} 根手指限制约束。'
            if missing:
                status += f'缺少 {len(missing)} 根：' + '、'.join(missing[:4])
            props.finger_limit_status = status

        contacts = score.get('score', {}).get('contacts') or []
        details = []
        for label, row in (('左脚', contacts[0] if len(contacts) > 0 else None),
                           ('右脚', contacts[1] if len(contacts) > 1 else None)):
            if row:
                details.append(f"{label}接触平均 {row.get('position_mean_mm', 0):.3f} mm / "
                               f"p95 {row.get('position_p95_mm', 0):.3f} mm")
        source_text = ('RGB MoCap 工程 ' + project['kind']) if project else 'FlowHMR final.npz'
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
