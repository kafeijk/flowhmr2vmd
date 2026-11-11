"""External contact solver worker (runs with a numpy+scipy Python).

Portions adapted from RGB MoCap (MIT License).
"""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

ADDON = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ADDON / 'vendor'))


def fs_path(path):
    """Windows long-path prefix for FlowHMR output folders with long titles."""
    path = str(path)
    if os.name != 'nt' or path.startswith('\\\\?\\'):
        return path
    absolute = os.path.abspath(path)
    if len(absolute) > 259:
        return '\\\\?\\' + absolute
    return path

from mocap import solver as m
from mocap import calibration as calib
from mocap import measured
from mocap import stabilize as stab
from mocap import surface as msurface
from mocap import source_surface
from mocap import surface_roll
from mocap import bone_overlay
from mocap.foot_calibration import corrected_world
import mocap.foot_calibration as fc


def enforce_foot_spacing(fixed, rest, ratio=0.65):
    """把左右脚踝的水平间距抬到模型静止脚距的 ratio 倍。

    只在双脚过度并拢（会互相穿模）的帧，绕髋部把整条腿向外旋转，
    保持腿部姿势与接触；返回修改的帧数。
    """
    points = fixed['points']
    world = fixed['world']
    min_sep = ratio * float(np.linalg.norm(rest[7][[0, 2]] - rest[8][[0, 2]]))
    separations = np.linalg.norm(
        (points[:, 7] - points[:, 8])[:, [0, 2]], axis=1)
    changed = 0
    for frame in np.flatnonzero(separations < min_sep):
        axis = (points[frame, 1] - points[frame, 2]).astype(np.float64)
        axis[1] = 0.0
        length = float(np.linalg.norm(axis))
        if length < 1e-9:
            continue
        axis /= length
        for side, (upper, lower, ankle) in enumerate(((1, 4, 7), (2, 5, 8))):
            sign = 1.0 if side == 0 else -1.0
            pad = (min_sep - separations[frame]) * 0.5 * sign
            hip = points[frame, upper]
            before = points[frame, ankle] - hip
            after = points[frame, ankle] + axis * pad - hip
            rotation = m.align(before, after)
            for joint in (upper, lower, ankle):
                world[frame, joint] = rotation @ world[frame, joint]
        changed += 1
    if changed:
        local = m.local_from_world(world)
        positions, rotations = m.fk(local, rest, fixed['root'])
        fixed['local'] = local
        fixed['world'] = rotations
        fixed['points'] = positions
    return changed


def build_evidence(raw_path, points, native, neutral_rest, vertices, fps):
    """按原项目 source.prepare 的方式生成腿部表面证据 F / F0。"""
    raw = np.load(fs_path(raw_path), mmap_mode='r')
    try:
        endpoints = source_surface.native_endpoints(raw)
    finally:
        raw._mmap.close()
    fitted = surface_roll.confidence_solve(
        points, native, neutral_rest, endpoints, fps)[0]
    endpoints0 = msurface.endpoints(vertices, 'smplh')[None]
    identity = np.tile(np.eye(3), (1, 22, 1, 1))
    fitted0 = surface_roll.confidence_solve(
        neutral_rest[None], identity, neutral_rest, endpoints0, 30)[0][0]
    frames = []
    neutral_frames = []
    for h, k in ((1, 4), (2, 5)):
        basis = bone_overlay.reference_frame(neutral_rest[k] - neutral_rest[h])
        frames.append(fitted[:, h] @ basis)
        neutral_frames.append(fitted0[h] @ basis)
    return np.stack(frames, 1), np.stack(neutral_frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        request = np.load(args.request, allow_pickle=True)
        names = json.loads(str(request['names_json']))
        source_names = names['source']
        target_names = names['target']
        fps = float(request['fps'])
        active = request['active'].astype(bool)
        points = request['source_points'].astype(np.float64)
        world = request['source_world'].astype(np.float64)
        rest = request['source_rest'].astype(np.float64)
        neutral = (request['source_neutral'].astype(np.float64)
                   if 'source_neutral' in request.files else None)
        target_rest = request['target_rest'].astype(np.float64)
        lock_toes = bool(request['lock_toes'])
        source_mode = str(request['source_mode']) if 'source_mode' in request.files else 'project'
        use_project_pt = bool(request['use_project_pt']) if 'use_project_pt' in request.files else False
        raw_path = (str(request['project_raw_path'])
                    if 'project_raw_path' in request.files else '')
        origin = (request['source_origin'].astype(np.float64)
                  if 'source_origin' in request.files else np.zeros(3))
        pt_fallback = ''

        # rig.npy stores float32 world matrices; project them onto SO(3)
        # exactly like RGB MoCap's source.prepare does.
        world = R.from_matrix(world.reshape(-1, 3, 3)).as_matrix().reshape(
            len(world), 22, 3, 3)
        model = np.load(fs_path(str(request['project_model_path'])), allow_pickle=True)
        betas = np.zeros(16, np.float64)
        native = world.copy()
        if use_project_pt:
            try:
                import torch
                params = torch.load(fs_path(str(request['project_pt_path'])),
                                    map_location='cpu', weights_only=False)
                betas = (params['shapes'].detach().cpu().numpy()
                         .reshape(-1)[:16].astype(np.float64))
                rot6d = (params['rot6d'].detach().cpu().numpy()
                         .reshape(len(world), 52, 3, 2).astype(np.float64))
                u = rot6d[..., 0]
                v = rot6d[..., 1] - (u * rot6d[..., 1]).sum(-1, keepdims=True) * u
                v /= np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
                w = np.cross(u, v)
                native = np.stack([u, v, w], -1)[:, :22]
                for joint in range(1, 22):
                    parent = m.PARENTS[joint]
                    if parent >= 0:
                        native[:, joint] = native[:, parent] @ native[:, joint]
            except Exception as exc:
                pt_fallback = str(exc)
                betas = np.zeros(16, np.float64)
                native = world.copy()
        vertices = (model['v_template'] +
                    np.einsum('vck,k->vc', model['shapedirs'][:, :, :16], betas))
        weights = model['weights']
        neutral_rest = (model['J_regressor'] @ vertices)[:22]

        if lock_toes:
            world[:, 10] = world[:, 7]
            world[:, 11] = world[:, 8]
            native[:, 10] = native[:, 7]
            native[:, 11] = native[:, 8]

        geo = {}
        for key in request.files:
            if key.startswith(('foot_', 'heel', 'toe', 'palm_')):
                geo[key] = request[key]

        os.environ['STOP_STAGNATION'] = '1'
        m.OUT = m.PREVIOUS = output
        m.LABEL = 'motion'
        m.FPS = fps
        fc.AUDIT = output
        fc.SOURCES.clear()

        np.savez(output / 'motion-source-neutral.npz',
                 vertices=vertices, weights=weights,
                 rest=neutral_rest, native_world=native)

        source = dict(points=points, world=world, rest=rest,
                      names=np.array(source_names), fps=fps)
        rest_baseline = rest.copy()
        head_length = float(np.linalg.norm(rest[15] - rest[12]))
        rest_baseline[15] = rest[12] + np.array([0., head_length, 0.])
        local = m.baseline(world, rest_baseline, target_rest)
        local, calibration = corrected_world(local, m.PARENTS, 'motion', geo)
        ratio = float(
            np.mean(np.linalg.norm(target_rest[[4, 5]] - target_rest[[1, 2]], axis=1) +
                    np.linalg.norm(target_rest[[7, 8]] - target_rest[[4, 5]], axis=1)) /
            np.mean(np.linalg.norm(rest[[4, 5]] - rest[[1, 2]], axis=1) +
                    np.linalg.norm(rest[[7, 8]] - rest[[4, 5]], axis=1)))
        root = points[:, 0, :] * ratio
        root[:, 1] += target_rest[0, 1] - rest[0, 1] * ratio
        positions, rotations = m.fk(local, target_rest, root)
        raw = dict(local=local, root=root, points=positions, world=rotations,
                   rest=target_rest, names=np.array(target_names),
                   ratio=ratio, **calibration)
        # 与原项目 pipeline 一致：先用 native 旋转重建身体链，再用 FBX 静息
        # 参考系重新基准化腿部，最后才做接触修正。
        base, _seed = m.characterize(raw, raw, source, native)
        if neutral is not None:
            want = calib.desired(world, neutral, target_rest)
            base['world'][:, [1, 2, 4, 5]] = want[:, [1, 2, 4, 5]]
        base['local'] = m.local_from_world(base['world'])
        base['points'], base['world'] = m.fk(base['local'], target_rest,
                                             base['root'])
        reference_points = base['points'].copy()
        safe_cache = {}
        original_safe = m.source_safe_reach
        original_two = m.two_bone

        def safe_reach(rest_value, src, side):
            return original_safe(rest_value, dict(
                src, points=reference_points, _safe_reach=safe_cache), side)

        def two_bone(base_data, root_value, goals, planned, src):
            return original_two(base_data, root_value, goals, planned,
                                dict(src, world=base_data['world']))

        m.source_safe_reach = safe_reach
        m.two_bone = two_bone
        fixed, plan, diagnostic = m.refine(base, source, active, geo)

        label = 'motion'
        kind = 'motion'
        m.OUT = m.PREVIOUS = output
        m.LABEL = label
        np.savez(output / f'{label}-source.npz',
                 names=np.array(source_names), parents=m.PARENTS,
                 rest=rest, points=points, world=world,
                 local=m.local_from_world(world), origin=origin,
                 fps=np.array(float(fps)))
        np.savez(output / f'{label}-source-neutral.npz',
                 vertices=vertices, weights=weights, rest=neutral_rest,
                 native_world=native)
        np.savez(output / f'{label}-source-contacts.npz',
                 active=active, fps=np.array(float(fps)))
        if neutral is not None:
            np.savez(output / f'{label}-neutral-calibration.npz', neutral=neutral)
        np.savez(output / f'{kind}-canonical.npz', rest=target_rest,
                 names=np.array(target_names))
        np.savez(output / f'{kind}-contact-geometry.npz', **geo)

        evidence = None
        if raw_path:
            evidence = build_evidence(
                raw_path, points, native, neutral_rest, vertices, fps)
            np.savez(output / f'{label}-evidence.npz',
                     frames=evidence[0], neutral_frames=evidence[1])
        np.savez(output / f'{label}-{kind}-rotation.npz', **base)
        np.savez(output / f'{label}-{kind}-stable-contact.npz', **fixed)
        np.savez(output / f'{label}-{kind}-contact-plan.npz',
                 active=active, anchors=plan['anchors'], goals=plan['goals'],
                 support_weights=plan['weights'],
                 palm_ids_left=plan['palms'][0],
                 palm_ids_right=plan['palms'][1])

        current = fixed
        if evidence is not None:
            m.run(kind)
            current = dict(np.load(
                output / f'{label}-{kind}-stable-contact.npz'))
            if neutral is not None:
                current = calib.reorient_pose(
                    current, world, neutral, m.align_many, m.fk,
                    m.local_from_world)
            seed = dict(np.load(output / f'{label}-{kind}-root-base.npz'))
            current, _ = stab.solve(current, seed, active, {})
            if neutral is not None:
                current = calib.reorient_pose(
                    current, world, neutral, m.align_many, m.fk,
                    m.local_from_world)
            evidence_frames, neutral_frames = evidence
            poles, _frames = msurface.mapped_poles(
                current, evidence_frames, neutral_frames)
            current = msurface.rebuild(current, poles, _frames)
            leg_frames, deltas, angles = measured.target_frames(target_rest)
            w = current['world'].copy()
            for joint in (1, 2, 4, 5):
                w[:, joint] = w[:, joint] @ deltas[joint]
            local = m.local_from_world(w)
            predicted, _ = m.fk(local, target_rest, current['root'])
            free_delta = predicted[:, [7, 8]] - current['points'][:, [7, 8]]
            free_delta[:, :, 1] = 0.0
            fixed_delta = free_delta.copy()
            runs = []
            for side in (0, 1):
                for start, end in m.trajectory.runs(active[:, side]):
                    fixed_delta[start:end, side] = np.median(
                        free_delta[start:min(end, start + 5), side], axis=0)
                    runs.append(dict(foot=side, start=start, end=end))
            offsets, _ = m.trajectory.contact_offsets(
                np.zeros((len(active), 4, 3)), active[:, :2],
                fixed_delta, free_delta, np.zeros((len(active), 3)),
                runs, m.FPS, np.ones((len(active), 2)),
                ~active[:, :2].any(1))
            delta = free_delta + offsets
            goals = current['points'][:, [7, 8]] + delta
            mapped = np.stack([
                evidence_frames[:, s] @ neutral_frames[s].T @ leg_frames[h]
                for s, h in enumerate((1, 2))], 1)
            if max(abs(np.array(angles))) >= 0.05:
                direct, _ = measured.rebuild_at_goals(
                    current, goals, mapped)
                solved, _ = measured.solve(
                    direct, current, active,
                    dict(iterations=10, physical_sigma=.008,
                         contact_sigma=.00005, root_prior=.004,
                         root_limit=min(.015, .025 * ratio)),
                    goal_delta=goals - direct['points'][:, [7, 8]])
                axis = measured.unit(
                    solved['points'][:, [7, 8]] - solved['points'][:, [1, 2]])
                current = msurface.rebuild(
                    solved,
                    measured.unit(np.cross(mapped[:, :, :, 0], axis)),
                    mapped)
        fixed = current
        enforce_foot_spacing(fixed, target_rest)
        score = m.score(fixed, source, active, geo, plan)
        m.source_safe_reach = original_safe
        m.two_bone = original_two
        # 与原项目一致：把源坐标原点加回去，得到绝对骨架位置。
        stance_root = fixed['root'] + origin
        stance_points = fixed['points'] + origin
        np.savez(output / 'stance.npz', root=stance_root, world=fixed['world'],
                 local=fixed['local'], points=stance_points, rest=target_rest,
                 names=np.array(target_names), ratio=ratio)
        (output / 'score.json').write_text(json.dumps(dict(
            score=score, ratio=ratio, frames=int(len(root)),
            source_mode=source_mode, pt_fallback=pt_fallback,
            diagnostic=dict(
                unreachable_frames_per_part=diagnostic['unreachable_frames_per_part'],
                geometry_iterations=diagnostic['geometry_iterations'],
                early_stop=diagnostic['early_stop'])), ensure_ascii=False,
            default=float), encoding='utf8')
        print('SOLVED', json.dumps(dict(frames=int(len(root)), ratio=ratio,
                                        mode=source_mode)))
    except Exception:
        detail = traceback.format_exc()
        (output / 'error.txt').write_text(detail, encoding='utf8')
        print(detail, file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
