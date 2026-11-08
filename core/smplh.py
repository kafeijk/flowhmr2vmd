"""Find and read FlowHMR final.npz / SMPL-H model.npz, then run SMPL-H FK.

Adapted from RGB MoCap's source preparation and SMPL-H handling (MIT).
"""
import os
import json
from pathlib import Path
import numpy as np

from .constants import NAMES, PARENTS


class SourceError(Exception):
    pass


def _candidate_models(folder, gender):
    folder = Path(folder)
    names = [gender, 'neutral']
    seen = set()
    env = os.environ.get('RGBMOCAP_SMPLH_MODEL')
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
                if path.is_file() and path not in seen:
                    seen.add(path)
                    yield path


def find_smplh_model(folder, gender):
    for path in _candidate_models(folder, gender):
        if path.is_file():
            return path
    # Bounded search for a SMPL-H model close to the output folder.
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
    if direct.is_file():
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


def load_final(path):
    with np.load(path, allow_pickle=True) as data:
        required = ('poses', 'trans', 'betas', 'foot_contacts', 'mocap_framerate')
        for key in required:
            if key not in data:
                raise SourceError(f'final.npz 缺少数据：{key}')
        poses = np.asarray(data['poses'], np.float64)
        trans = np.asarray(data['trans'], np.float64)
        betas = np.asarray(data['betas'], np.float64).reshape(-1)
        contacts = np.asarray(data['foot_contacts'], np.float64)
        fps = float(data['mocap_framerate'])
        gender = str(data['gender']) if 'gender' in data else 'neutral'
    if poses.ndim != 3 or poses.shape[1] < 52 or poses.shape[2] != 3:
        raise SourceError('final.npz 的 poses 必须是 (帧, 52, 3) 轴角。')
    if trans.shape != (len(poses), 3):
        raise SourceError('final.npz 的 trans 形状不正确。')
    if contacts.shape[0] != len(poses) or contacts.shape[1] < 4:
        raise SourceError('final.npz 的 foot_contacts 形状不正确。')
    if len(poses) < 2:
        raise SourceError('动作至少需要两帧。')
    if not 1 <= fps <= 240:
        raise SourceError('源动作帧率无效。')
    return dict(poses=poses[:, :52], trans=trans, betas=betas[:16],
                contacts=contacts[:, :4], fps=fps, gender=gender,
                frames=len(poses), path=path)


def load_model(path):
    with np.load(path, allow_pickle=True) as data:
        required = ('v_template', 'shapedirs', 'J_regressor', 'weights',
                    'kintree_table')
        for key in required:
            if key not in data:
                raise SourceError(f'SMPL-H model.npz 缺少数据：{key}')
        model = {key: np.asarray(data[key]) for key in required}
    if model['J_regressor'].shape[0] < 52:
        raise SourceError('SMPL-H model.npz 的 J_regressor 不足 52 个关节。')
    return model


def rotvec_to_matrix(rotvec):
    """Vectorised Rodrigues formula for (..., 3) axis-angle."""
    rotvec = np.asarray(rotvec, np.float64)
    theta = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    axis = rotvec / np.maximum(theta, 1e-12)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = np.zeros_like(x)
    K = np.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1)
    K = K.reshape(rotvec.shape[:-1] + (3, 3))
    eye = np.broadcast_to(np.eye(3), K.shape)
    angle = theta[..., None]
    return eye + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def forward_kinematics(final, model):
    poses = final['poses']
    trans = final['trans']
    betas = final['betas']
    n = len(poses)
    vertices = (model['v_template'] +
                np.einsum('vck,k->vc', model['shapedirs'][:, :, :16], betas))
    joints = model['J_regressor'] @ vertices
    parents = model['kintree_table'][0].astype(np.int64).copy()
    parents[0] = -1
    local = rotvec_to_matrix(poses)
    world = np.empty((n, 52, 3, 3), np.float64)
    points = np.empty((n, 52, 3), np.float64)
    world[:, 0] = local[:, 0]
    points[:, 0] = trans
    for joint in range(1, 52):
        parent = parents[joint]
        world[:, joint] = world[:, parent] @ local[:, joint]
        points[:, joint] = points[:, parent] + np.einsum(
            'tij,j->ti', world[:, parent], joints[joint] - joints[parent])
    return dict(world=world, points=points, joints=joints, parents=parents,
                vertices=vertices, weights=model['weights'],
                fps=final['fps'], gender=final['gender'])


def clean_contacts(contacts):
    left = np.maximum(contacts[:, 0], contacts[:, 1]) > .7
    right = np.maximum(contacts[:, 2], contacts[:, 3]) > .7

    def clean(mask, close=2, minimum=3):
        mask = np.asarray(mask, bool).copy()
        index = 0
        while index < len(mask):
            if mask[index]:
                index += 1
                continue
            end = index
            while end < len(mask) and not mask[end]:
                end += 1
            if index > 0 and end < len(mask) and end - index <= close:
                mask[index:end] = True
            index = end
        index = 0
        while index < len(mask):
            if not mask[index]:
                index += 1
                continue
            end = index
            while end < len(mask) and mask[end]:
                end += 1
            if end - index < minimum:
                mask[index:end] = False
            index = end
        return mask

    active = np.zeros((len(contacts), 4), bool)
    active[:, 0] = clean(left)
    active[:, 1] = clean(right)
    return active


_FINGER_IDS = {
    'Left': {'Thumb': [34, 35, 36], 'Index': [22, 23, 24],
             'Middle': [25, 26, 27], 'Ring': [31, 32, 33],
             'Pinky': [28, 29, 30]},
    'Right': {'Thumb': [49, 50, 51], 'Index': [37, 38, 39],
              'Middle': [40, 41, 42], 'Ring': [46, 47, 48],
              'Pinky': [43, 44, 45]},
}


def prepare_source(final, fkdata, lock_toes=False):
    """Build solver inputs and reordered 52-joint data for finger baking."""
    world = fkdata['world']
    points = fkdata['points']
    joints = fkdata['joints']
    if lock_toes:
        world[:, 10] = world[:, 7]
        world[:, 11] = world[:, 8]
    origin = points[0, 0].copy()
    relative = points - origin
    rest = joints[:22] - joints[0]

    perm = list(range(22))
    fnames = list(NAMES)
    for side in ('Left', 'Right'):
        for finger in ('Thumb', 'Index', 'Middle', 'Ring', 'Pinky'):
            for phalanx in (1, 2, 3):
                perm.append(_FINGER_IDS[side][finger][phalanx - 1])
                fnames.append(side + 'Hand' + finger + str(phalanx))
    perm = np.asarray(perm, np.int64)
    frames52 = np.zeros((len(world), 52, 4, 4), np.float32)
    frames52[:, :, :3, :3] = world[:, perm]
    frames52[:, :, :3, 3] = points[:, perm]
    frames52[:, :, 3, 3] = 1.0
    rest52 = joints[perm] - joints[0]
    ends52 = np.zeros((52, 3))
    for index in range(22, 52):
        phalanx = (index - 22) % 3 + 1
        if phalanx == 3:
            direction = rest52[index] - rest52[index - 1]
            norm = np.linalg.norm(direction)
            ends52[index] = rest52[index] + direction / max(norm, 1e-9) * .01

    return dict(
        world=world[:, :22], points=relative[:, :22], rest=rest,
        neutral_vertices=fkdata['vertices'], neutral_weights=fkdata['weights'],
        native_world=world[:, :22], active=clean_contacts(final['contacts']),
        fps=final['fps'], origin=origin, frames52=frames52, rest52=rest52,
        ends52=ends52, fnames=fnames, source_names=list(NAMES),
    )


def find_project(folder):
    """Locate an RGB MoCap project's prepared corrected/initial source."""
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
        data = json.loads(metadata_path.read_text(encoding='utf8'))
    except Exception:
        return None
    base = metadata_path.parent
    kind = data.get('kind', 'corrected_initial')
    frames_file = data.get('frames_file')
    asset_file = data.get('asset_file')
    if not frames_file or not asset_file:
        return None
    model_path = data.get('model_path')
    if not model_path or not Path(model_path).is_file():
        model_path = find_smplh_model(base, 'neutral')
    pt = base / f'{kind}.pt'
    return dict(
        metadata=data, folder=base, kind=kind,
        frames_file=(Path(frames_file) if os.path.isabs(frames_file)
                     else base / frames_file),
        asset_file=(Path(asset_file) if os.path.isabs(asset_file)
                    else base / asset_file),
        model_path=(Path(model_path) if model_path else None),
        contacts=(base / f'{kind}_contacts.npz'
                  if (base / f'{kind}_contacts.npz').is_file() else None),
        correction=(base / f'{kind}_correction.json'
                    if (base / f'{kind}_correction.json').is_file() else None),
        pt=(pt if pt.is_file() else None),
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
        raise SourceError('未找到 RGB MoCap 工程的动作缓存。')
    if project['model_path'] is None:
        raise SourceError('未找到工程使用的 SMPL-H model.npz。')
    frames = np.asarray(np.load(project['frames_file'], mmap_mode='r'),
                        np.float64)
    if frames.ndim != 4 or frames.shape[1] < 52 or frames.shape[2:] != (4, 4):
        raise SourceError('工程 rig.npy 必须是 (帧, 52, 4, 4)。')
    asset = np.load(project['asset_file'])
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
            correction = json.loads(project['correction'].read_text(
                encoding='utf8'))
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
        if raw_path.is_file():
            raw_file = str(raw_path)
    active = np.zeros((n, 4), bool)
    if project['contacts'] is not None:
        contacts = np.load(project['contacts'])
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
            kind=project['kind'], floor=floor, origin=origin.tolist()),
    )


def find_source(folder):
    final = find_final_npz(folder)
    if final is not None:
        return 'final', final
    project = find_project(folder)
    if project is not None:
        return 'project', project
    return None, None
