import numpy as np
from scipy.spatial.transform import Rotation as R
from . import surface
from .bounded_stabilizer import solve
from .calibration import frame
m=surface.m
unit=surface.unit
def target_frames(rest):
    canonical = frame([0.0, -1.0, 0.0])
    frames = {}
    deltas = {}
    angles = []
    for (j, ch) in [(1, 4), (2, 5), (4, 7), (5, 8)]:
        d = rest[ch] - rest[j]
        coronal = np.array([d[0], -np.hypot(d[1], d[2]), 0.0])
        frames[j] = frame(coronal)
        D = frame(d)
        deltas[j] = D @ canonical.T @ frames[j] @ D.T
        angles.append(float(np.degrees(np.arctan2(coronal[0], -coronal[1]))))
    return (frames, deltas, angles)

def rebuild_at_goals(old, goals, frames):
    p = old['points']
    w = old['world'].copy()
    errs = []
    for (s, (h, k, a)) in enumerate([(1, 4, 7), (2, 5, 8)]):
        direction = unit(goals[:, s] - p[:, h])
        distance = np.linalg.norm(goals[:, s] - p[:, h], axis=-1)
        u = np.linalg.norm(old['rest'][k] - old['rest'][h])
        v = np.linalg.norm(old['rest'][a] - old['rest'][k])
        d = np.clip(distance, abs(u - v) + 1e-07, u + v - 1e-07)
        pol = unit(np.cross(frames[:, s, :, 0], direction))
        along = (u * u - v * v + d * d) / (2 * d)
        height = np.sqrt(np.maximum(0, u * u - along * along))
        knee = p[:, h] + direction * along[:, None] + pol * height[:, None]
        end = p[:, h] + direction * d[:, None]
        for (j, ch, nv) in [(h, k, knee - p[:, h]), (k, a, end - knee)]:
            w[:, j] = m.align_many(p[:, ch] - p[:, j], nv) @ w[:, j]
        errs.append(np.abs(distance - d))
    local = m.local_from_world(w)
    (pp, ww) = m.fk(local, old['rest'], old['root'])
    return (dict(old, points=pp, world=ww, local=local), np.stack(errs, 1))

def rotation_step(d):
    return float(np.degrees(R.from_matrix((d['local'][:-1, [1, 2, 4, 5]].swapaxes(-1, -2) @ d['local'][1:, [1, 2, 4, 5]]).reshape(-1, 3, 3)).magnitude()).max())