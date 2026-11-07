from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
SOURCES={}
def frame(vertices, heel, toe):
    patch = vertices[np.unique(np.r_[heel, toe])]
    centre = patch.mean(0)
    (_, s, V) = np.linalg.svd(patch - centre, full_matrices=False)
    if s[1] < 1e-06:
        raise ValueError('Degenerate registered sole reference')
    up = V[-1]
    if up[1] < 0:
        up = -up
    forward = vertices[toe].mean(0) - vertices[heel].mean(0)
    forward -= up * (up @ forward)
    forward /= np.linalg.norm(forward)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    basis = np.column_stack([right, up, forward])
    assert np.linalg.det(basis) > 0.999999
    return basis

def source_reference(label):
    if label not in SOURCES:
        a = np.load(AUDIT / f'{label}-source-neutral.npz')
        v = a['vertices']
        w = a['weights']
        rest = a['rest']
        frames = []
        ids = []
        for side in [0, 1]:
            ankle = 7 + side
            toe = 10 + side
            eligible = np.flatnonzero((w[:, [ankle, toe]].sum(1) > 0.7) & (v[:, 1] < rest[ankle, 1] + 0.02))
            floor = np.percentile(v[eligible, 1], 1)
            bottom = eligible[v[eligible, 1] < floor + 0.006]
            forward = rest[toe] - rest[ankle]
            forward[1] = 0.0
            forward /= np.linalg.norm(forward)
            along = (v[bottom] - rest[ankle]) @ forward
            middle = np.median(along)
            heel = bottom[along <= middle]
            fore = bottom[along > middle]
            frames.append(frame(v, heel, fore))
            ids.append([heel, fore])
        SOURCES[label] = (a['native_world'], np.array(frames), ids)
    return SOURCES[label]

def corrected_world(old_local, parents, label, geo):
    world = old_local.copy()
    for (j, p) in enumerate(parents):
        if p >= 0:
            world[:, j] = world[:, p] @ world[:, j]
    before = world.copy()
    (native, source_frames, ids) = source_reference(label)
    calibration = []
    target_frames = []
    for side in [0, 1]:
        target = frame(geo[f'foot_v{side}'], geo[f'heel{side}'], geo[f'toe{side}'])
        C = source_frames[side] @ target.T
        target_frames.append(target)
        calibration.append(C)
        for j in [7 + side, 10 + side]:
            world[:, j] = native[:, j] @ C
    unchanged = [j for j in range(22) if j not in [7, 8, 10, 11]]
    assert abs(world[:, unchanged] - before[:, unchanged]).max() < 1e-10
    local = world.copy()
    for (j, p) in enumerate(parents):
        if p >= 0:
            local[:, j] = world[:, p].swapaxes(-1, -2) @ world[:, j]
    C = np.array(calibration)
    assert np.max(abs(C.swapaxes(-1, -2) @ C - np.eye(3))) < 1e-10
    return (local, dict(foot_basis=C, source_sole_frames=source_frames, target_sole_frames=np.array(target_frames), foot_calibration_version=2))