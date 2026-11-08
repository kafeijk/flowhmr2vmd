import numpy as np
from scipy.spatial.transform import Rotation as R
from .stabilize import m,unit,flex
from .calibration import frame,reference
from mocap.surface_landmarks import PATCHES
from mocap.surface_roll import confidence_solve
from mocap.bone_overlay import reference_frame
def endpoints(v, topology):
    return np.stack([np.stack([v[..., p[k], :].mean(-2) for k in ['front', 'back', 'side_a', 'side_b']], -2) for p in PATCHES[topology]], -3)

def pole(p, h, k, a):
    axis = unit(p[:, a] - p[:, h])
    return unit(p[:, k] - p[:, h] - axis * np.sum((p[:, k] - p[:, h]) * axis, axis=-1)[:, None])

def mapped_poles(data, F, F0):
    poles = []
    frames = []
    for (s, (h, k, a)) in enumerate([(1, 4, 7), (2, 5, 8)]):
        target_rest = frame(data['rest'][k] - data['rest'][h])
        target_neutral = reference(data['rest'])[h] @ target_rest
        ft = F[:, s] @ F0[s].T @ target_neutral
        axis = unit(data['points'][:, a] - data['points'][:, h])
        candidate = np.cross(ft[:, :, 0], axis)
        poles.append(unit(candidate))
        frames.append(ft)
    return (np.stack(poles, 1), np.stack(frames, 1))

def rebuild(data, poles, frames):
    p = data['points']
    w = data['world'].copy()
    for (s, (h, k, a)) in enumerate([(1, 4, 7), (2, 5, 8)]):
        axis = unit(p[:, a] - p[:, h])
        length = np.linalg.norm(p[:, a] - p[:, h], axis=1)
        u = np.linalg.norm(data['rest'][k] - data['rest'][h])
        v = np.linalg.norm(data['rest'][a] - data['rest'][k])
        along = (u * u - v * v + length * length) / (2 * length)
        height = np.sqrt(np.maximum(0, u * u - along * along))
        middle = p[:, h] + axis * along[:, None] + poles[:, s] * height[:, None]
        for (j, ch, newvec) in [(h, k, middle - p[:, h]), (k, a, p[:, a] - middle)]:
            w[:, j] = m.align_many(p[:, ch] - p[:, j], newvec) @ w[:, j]
            y = unit(newvec)
            D = frame(data['rest'][ch] - data['rest'][j])
            have = w[:, j] @ D[:, 0]
            want = unit(frames[:, s, :, 0] - y * np.sum(y * frames[:, s, :, 0], axis=-1)[:, None])
            angle = np.arctan2(np.sum(y * np.cross(have, want), axis=-1), np.sum(have * want, axis=-1))
            w[:, j] = R.from_rotvec(y * angle[:, None]).as_matrix() @ w[:, j]
    local = m.local_from_world(w)
    (points, world) = m.fk(local, data['rest'], data['root'])
    return dict(data, local=local, points=points, world=world)

def angle(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(unit(a) * unit(b), axis=-1), -1, 1)))