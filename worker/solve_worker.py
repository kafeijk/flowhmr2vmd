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

from mocap import solver as m
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
        target_rest = request['target_rest'].astype(np.float64)
        lock_toes = bool(request['lock_toes'])
        source_mode = str(request['source_mode']) if 'source_mode' in request.files else 'final'
        use_project_pt = bool(request['use_project_pt']) if 'use_project_pt' in request.files else False
        pt_fallback = ''

        if source_mode == 'project':
            # rig.npy stores float32 world matrices; project them onto SO(3)
            # exactly like RGB MoCap's source.prepare does.
            world = R.from_matrix(world.reshape(-1, 3, 3)).as_matrix().reshape(
                len(world), 22, 3, 3)
            model = np.load(str(request['project_model_path']), allow_pickle=True)
            betas = np.zeros(16, np.float64)
            native = world.copy()
            if use_project_pt:
                try:
                    import torch
                    params = torch.load(str(request['project_pt_path']),
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
        else:
            native = request['native_world'].astype(np.float64)
            vertices = request['neutral_vertices']
            weights = request['neutral_weights']
            neutral_rest = rest

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
        base = dict(local=local, root=root, points=positions, world=rotations,
                    rest=target_rest, names=np.array(target_names),
                    ratio=ratio, **calibration)
        fixed, plan, diagnostic = m.refine(base, source, active, geo)
        enforce_foot_spacing(fixed, target_rest)
        if source_mode == 'project':
            # rig.npy 的关节世界姿态沿骨段建立坐标系，与 final.npz 的纯动作
            # 增量不同约定。直接套用静止段方向会把头部的静息前倾当成动作
            # 偏移，导致角色低头；这里改用“头相对颈部”的旋转驱动目标头部。
            relative = world[:, 15] @ np.transpose(world[:, 12], (0, 2, 1))
            fixed['world'][:, 15] = fixed['world'][:, 12] @ relative
        score = m.score(fixed, source, active, geo, plan)
        np.savez(output / 'stance.npz', root=fixed['root'], world=fixed['world'],
                 local=fixed['local'], points=fixed['points'], rest=target_rest,
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
