"""Locate and read the FlowHMR / RGB MoCap (toyxyz) project cache.

`final.npz` is only used as the output-folder marker. The action comes from
the prepared project cache described by corrected_initial_mixamo.json
(rig.npy / rig-asset.npz / surface.npy / *.pt).
"""
import json
import os
from pathlib import Path

import numpy as np


class SourceError(Exception):
    pass


def _fs_path(path):
    """Windows long-path prefix.

    Blender's Python is not long-path aware, so paths over 260 characters
    (common for FlowHMR output folders with long Korean titles) cannot be
    opened without the ``\\\\?\\`` prefix.
    """
    path = str(path)
    if os.name != 'nt' or path.startswith('\\\\?\\'):
        return path
    absolute = os.path.abspath(path)
    if len(absolute) > 259:
        return '\\\\?\\' + absolute
    return path


def _is_file(path):
    return os.path.isfile(_fs_path(path))


def _candidate_models(folder, gender):
    folder = Path(folder)
    names = [gender, 'neutral']
    seen = set()
    env = os.environ.get('FLOWHMR2VMD_SMPLH_MODEL')
    if env:
        yield Path(env)
    for depth in range(0, 8):
        base = folder
        for _ in range(depth):
            base = base.parent
        for rel in ('app/flowhmr/assets/body_models/smplh',
                    'assets/body_models/smplh',
                    'body_models/smplh',
                    'models/body_models/smplh'):
            for name in names:
                path = base / rel / name / 'model.npz'
                if _is_file(path) and path not in seen:
                    seen.add(path)
                    yield path


def find_smplh_model(folder, gender):
    for path in _candidate_models(folder, gender):
        if _is_file(path):
            return path
    count = 0
    folder = Path(folder)
    for root, dirs, files in os.walk(folder):
        count += 1
        if count > 4000:
            break
        lowered = root.lower()
        if 'smplh' not in lowered:
            continue
        if 'model.npz' in files:
            return Path(root) / 'model.npz'
    return None


def find_final_npz(folder):
    folder = Path(folder)
    direct = folder / 'final.npz'
    if _is_file(direct):
        return direct
    found = []
    for root, dirs, files in os.walk(folder):
        if 'final.npz' in files:
            path = Path(root) / 'final.npz'
            found.append((len(path.parts), -path.stat().st_mtime, path))
    if not found:
        return None
    found.sort(key=lambda item: (item[0], item[1]))
    return found[0][2]


def find_project(folder):
    """Locate an RGB MoCap / toyxyz project's prepared corrected source."""
    folder = Path(folder)
    candidates = []
    for root, dirs, files in os.walk(folder):
        for name, priority in (('corrected_initial_mixamo.json', 0),
                               ('initial_mixamo.json', 1)):
            if name in files:
                path = Path(root) / name
                candidates.append((priority, len(path.parts), path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    metadata_path = candidates[0][2]
    try:
        with open(_fs_path(metadata_path), encoding='utf8') as handle:
            data = json.load(handle)
    except Exception:
        return None
    base = metadata_path.parent
    kind = data.get('kind', 'corrected_initial')
    frames_file = data.get('frames_file')
    asset_file = data.get('asset_file')
    if not frames_file or not asset_file:
        return None
    model_path = data.get('model_path')
    if not model_path or not _is_file(model_path):
        model_path = find_smplh_model(base, 'neutral')
    contacts = base / f'{kind}_contacts.npz'
    correction = base / f'{kind}_correction.json'
    pt = base / f'{kind}.pt'
    return dict(
        metadata=data, folder=base, kind=kind,
        frames_file=(Path(frames_file) if os.path.isabs(frames_file)
                     else base / frames_file),
        asset_file=(Path(asset_file) if os.path.isabs(asset_file)
                    else base / asset_file),
        model_path=(Path(model_path) if model_path else None),
        contacts=(contacts if _is_file(contacts) else None),
        correction=(correction if _is_file(correction) else None),
        pt=(pt if _is_file(pt) else None),
        fps=float(data.get('fps', 30.0)),
        frames=int(data.get('frames', 0)),
        names=list(data.get('names', [])))


def prepare_project(project):
    """Read the prepared project rig/contacts using the app's conventions.

    The prepared frame positions are absolute. Subtracting origin with the
    correction floor (not the full root height) keeps the source restitution
    and points in the same frame as RGB MoCap's source.prepare.
    """
    if project is None:
        raise SourceError('未找到 FlowHMR / RGB MoCap 工程的动作缓存。')
    if project['model_path'] is None:
        raise SourceError('未找到工程使用的 SMPL-H model.npz。')
    frames_file = project['frames_file']
    if not _is_file(frames_file):
        raise SourceError(f'工程缺少 rig.npy：{frames_file}')
    asset_file = project['asset_file']
    if not _is_file(asset_file):
        raise SourceError(f'工程缺少 rig-asset.npz：{asset_file}')
    frames = np.asarray(np.load(_fs_path(frames_file), mmap_mode='r'),
                        np.float64)
    if frames.ndim != 4 or frames.shape[1] < 52 or frames.shape[2:] != (4, 4):
        raise SourceError('工程 rig.npy 必须是 (帧, 52, 4, 4)。')
    asset = np.load(_fs_path(asset_file))
    for key in ('rest', 'ends'):
        if key not in asset:
            raise SourceError(f'工程 rig-asset.npz 缺少 {key}。')
    names = list(project['names'])
    if len(names) < 52:
        raise SourceError('工程 mixamo.json 的骨骼名不足 52 个。')
    n = len(frames)
    fps = project['fps'] if 1 <= project['fps'] <= 240 else 30.0
    floor = 0.0
    if project['correction'] is not None:
        try:
            with open(_fs_path(project['correction']), encoding='utf8') as handle:
                correction = json.load(handle)
            floor = float(correction.get('floor', 0.0))
        except Exception:
            floor = 0.0
    world = frames[:, :22, :3, :3]
    points = frames[:, :22, :3, 3]
    origin = points[0, 0].copy()
    origin[1] = floor
    points = points - origin
    rest = np.asarray(asset['rest'], np.float64)[:22]
    rest52 = np.asarray(asset['rest'], np.float64)
    ends52 = np.asarray(asset['ends'], np.float64)
    neutral = None
    if 'fbx_rest_frames' in asset and 'fbx_display_frames' in asset:
        neutral = (np.asarray(asset['fbx_rest_frames'][:22], np.float64) @
                   np.swapaxes(
                       np.asarray(asset['fbx_display_frames'][:22], np.float64),
                       -1, -2))
    raw_file = None
    raw_name = project['metadata'].get('raw_file')
    if isinstance(raw_name, str):
        raw_path = Path(raw_name) if Path(raw_name).is_absolute() else (
            Path(project['folder']) / raw_name)
        if _is_file(raw_path):
            raw_file = str(raw_path)
    active = np.zeros((n, 4), bool)
    if project['contacts'] is not None:
        contacts = np.load(_fs_path(project['contacts']))
        if 'segments' in contacts:
            for row in contacts['segments']:
                side = int(round(float(row[0])))
                start = int(round(float(row[1])))
                end = int(round(float(row[2])))
                if 0 <= side < 2:
                    active[max(0, start):min(n, end), side] = True
    return dict(
        world=world, points=points, rest=rest, active=active, fps=fps,
        frames52=frames.astype(np.float32), rest52=rest52, ends52=ends52,
        fnames=names, source_names=names[:22], neutral=neutral,
        raw_file=raw_file,
        project=dict(
            model_path=str(project['model_path']),
            pt_path=str(project['pt']) if project['pt'] is not None else '',
            use_project_pt=bool(project['pt'] is not None),
            kind=project['kind'], floor=floor, origin=origin.tolist()))
