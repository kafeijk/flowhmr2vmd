"""Explicit source bind removal and a virtual common leg reference.

Adapted from toyxyz RGB MoCap (MIT License), mocap/retarget/calibration.py.
The character's skin bind, hip pivots, segment lengths and source caches
remain untouched. A virtual reference rotates the leg segments to parallel
down axes; the pelvis width is still the actual character's width.
"""
import numpy as np

CHAINS = [(1, 4, 7), (2, 5, 8)]
LEGS = [1, 2, 4, 5]


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def frame(direction, front=np.array([0., 0., 1.])):
    y = unit(np.asarray(direction, float))
    z = unit(front - y * np.dot(y, front))
    x = unit(np.cross(y, z))
    return np.stack([x, y, np.cross(x, y)], axis=-1)


def reference(rest):
    world = np.tile(np.eye(3), (22, 1, 1))
    canonical = frame([0., -1., 0.])
    for h, k, a in CHAINS:
        world[h] = canonical @ frame(rest[k] - rest[h]).T
        world[k] = canonical @ frame(rest[a] - rest[k]).T
    return world


def desired(source_world, neutral, rest):
    common = reference(rest)
    out = source_world.copy()
    for j in LEGS:
        out[:, j] = source_world[:, j] @ neutral[j].T @ common[j]
    return out


def reorient_pose(data, source_world, neutral, align_many, fk,
                  local_from_world):
    """Change outgoing roll while keeping every existing joint centre fixed."""
    world = data['world'].copy()
    want = desired(source_world, neutral, data['rest'])
    for h, k, a in CHAINS:
        for j, ch in [(h, k), (k, a)]:
            segment = data['rest'][ch] - data['rest'][j]
            predicted = want[:, j] @ segment
            actual = data['points'][:, ch] - data['points'][:, j]
            world[:, j] = align_many(predicted, actual) @ want[:, j]
    local = local_from_world(world)
    points, world = fk(local, data['rest'], data['root'])
    assert np.max(np.abs(points - data['points'])) < 1e-7
    return dict(data, local=local, points=points, world=world)
