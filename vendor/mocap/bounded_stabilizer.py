from pathlib import Path
import os,time,json
import numpy as np
from scipy.sparse import coo_matrix,diags,vstack
from scipy.sparse.linalg import spsolve
from scipy.ndimage import gaussian_filter1d,maximum_filter1d
from scipy.spatial.transform import Rotation as R
from . import solver as m
from .calibration import reorient_pose
def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)

def flex(p):
    return np.arccos(np.clip(np.sum(unit(p[:, [4, 5]] - p[:, [1, 2]]) * unit(p[:, [7, 8]] - p[:, [4, 5]]), axis=-1), -1, 1))

def stable_pole(old, seed, side, env):
    (a, b, c) = m.CHAINS[side]
    p = old['points']
    sp = seed['points']
    n = len(p)
    axis = unit(p[:, c] - p[:, a])
    su = sp[:, b] - sp[:, a]
    sv = sp[:, c] - sp[:, b]
    sf = flex(sp)[:, side]
    reliable = sf > np.radians(40)
    if reliable.sum() < 5:
        reliable = sf >= np.percentile(sf, 85)
    normal = unit(np.median(np.einsum('tji,tj->ti', seed['world'][:, 0], unit(np.cross(su, sv)))[reliable], axis=0))
    if np.linalg.norm(normal) < 1e-06:
        normal = np.array([1.0, 0.0, 0.0])
    cross = np.cross(axis, seed['world'][:, 0] @ normal)
    bad = np.linalg.norm(cross, axis=1) < 1e-06
    if bad.any():
        alternatives = np.cross(axis[bad, None], seed['world'][bad, 0].swapaxes(-1, -2))
        cross[bad] = alternatives[np.arange(bad.sum()), np.argmax(np.linalg.norm(alternatives, axis=-1), axis=1)]
    prior = unit(cross)
    observed = unit(p[:, b] - p[:, a] - axis * np.sum((p[:, b] - p[:, a]) * axis, axis=1)[:, None])
    raw = np.arctan2(np.sum(axis * np.cross(prior, observed), 1), np.sum(prior * observed, 1))
    angle = np.degrees(flex(p)[:, side])
    valid = angle > 40
    ids = np.flatnonzero(valid)
    phase = np.interp(np.arange(n), ids, np.unwrap(raw[valid])) if len(ids) else np.zeros(n)
    observed_phase = raw + 2 * np.pi * np.round((phase - raw) / (2 * np.pi))
    weight = np.clip(angle / 45, 0, 1) ** 4
    weight[angle < 5] = 0
    D = diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n)).tocsr()
    temporal = 2.0 * (1 - np.minimum.reduce([weight[:-2], weight[1:-1], weight[2:]])) + 0.02
    fit = spsolve(diags(weight + 1e-05) + D.T @ diags(temporal) @ D, weight * observed_phase + 1e-05 * phase)
    corrected = phase + env * (fit - phase)
    return R.from_rotvec(axis * corrected[:, None]).apply(prior)

def solve(old, seed, active, cfg, goal_delta=None):
    start = time.perf_counter()
    p = old['points']
    rest = old['rest']
    n = len(p)
    hands = active[:, 2:].any(1) if active.shape[1] > 2 else np.zeros(len(active), bool)
    active = active[:, :2]
    theta0 = flex(p)
    ref = flex(seed['points'])
    root0 = old['root']
    goals = p[:, [7, 8]].copy()
    ext = maximum_filter1d(active.astype(float), 13, axis=0, mode='nearest')
    env = gaussian_filter1d(ext, 1.5, axis=0, truncate=3, mode='nearest')
    env[active] = 1
    env[env < 1e-06] = 0
    enabled = env > 0
    rf = env.max(1) * (1 - gaussian_filter1d(maximum_filter1d(hands.astype(float), 13, mode='nearest'), 1.5, mode='nearest'))
    lengths = np.array([[np.linalg.norm(rest[b] - rest[a]), np.linalg.norm(rest[c] - rest[b])] for (a, b, c) in m.CHAINS[:2]])
    if goal_delta is not None:
        goals += goal_delta * env[:, :, None]
    (u, v) = lengths.T
    hipoff = p[:, [1, 2]] - root0[:, None]
    near = np.clip((45 - np.degrees(theta0)) / 35, 0, 1)
    near = near * near * (3 - 2 * near)
    speed = np.abs(np.gradient(ref, axis=0)) * m.FPS
    slow = 0.25 + 0.75 / (1 + (speed / 2.0) ** 2)
    w = env * (cfg.get('base_physical', 0.25) + (1 - cfg.get('base_physical', 0.25)) * near) * slow
    f = np.arange(n)
    parts = []
    targets = []

    def add(cols, coefs, target):
        cols = np.asarray(cols)
        coefs = np.broadcast_to(coefs, cols.shape)
        rr = np.repeat(np.arange(len(cols)), cols.shape[1])
        parts.append(coo_matrix((coefs.ravel(), (rr, cols.ravel())), shape=(len(cols), n * 5)).tocsr())
        targets.append(np.broadcast_to(target, (len(cols),)).copy())
    for j in range(3):
        add((f * 5 + j)[:, None], (1 / np.maximum(cfg.get('root_prior', 0.004) * rf, 1e-08))[:, None], 0)
    for s in range(2):
        add((f * 5 + 3 + s)[:, None], (1 / np.maximum(0.18 * env[:, s], 1e-08))[:, None], 0)
    d2 = np.array([1.0, -2.0, 1.0])
    frames = np.stack([f[:-2], f[1:-1], f[2:]], 1)
    for s in range(2):
        ew = np.maximum.reduce([env[:-2, s], env[1:-1, s], env[2:, s]])
        weight = ew / cfg.get('extra_sigma', 0.008)
        add(frames * 5 + 3 + s, d2 * weight[:, None], -np.diff(theta0[:, s] - ref[:, s], n=2) * weight)
        weight = np.sqrt(np.maximum.reduce([w[:-2, s], w[1:-1, s], w[2:, s]])) / cfg.get('physical_sigma', 0.004)
        add(frames * 5 + 3 + s, d2 * weight[:, None], -np.diff(theta0[:, s], n=2) * weight)
    for j in range(3):
        add(frames * 5 + j, d2 / cfg.get('root_sigma', 0.0006), -np.diff(root0[:, j] - seed['root'][:, j], n=2) / cfg.get('root_sigma', 0.0006) if cfg.get('root_delta', False) else 0)
    A = vstack(parts).tocsr()
    b = np.concatenate(targets)
    H = A.T @ A
    Ab = A.T @ b
    q = np.zeros((n, 5))
    costs = []
    I = np.eye(3)[None, None]

    def predict(q):
        hip = root0[:, None] + q[:, None, :3] + hipoff
        goal = goals + q[:, None, :3] * (1 - env[:, :, None])
        vec = goal - hip
        r = np.maximum(np.linalg.norm(vec, axis=-1), 1e-08)
        axis = vec / r[:, :, None]
        theta = theta0 + q[:, 3:]
        d = np.sqrt(np.maximum(u * u + v * v + 2 * u * v * np.cos(theta), 1e-10))
        end = hip + axis * d[:, :, None]
        return (end - goal, axis, r, d, theta)
    sigma = cfg.get('contact_sigma', 0.00015)
    cw = env / sigma

    def cost(q):
        (err, *_) = predict(q)
        return float(np.sum((err * cw[:, :, None]) ** 2) + np.sum((A @ q.ravel() - b) ** 2))
    for iteration in range(cfg.get('iterations', 10)):
        (err, axis, r, d, theta) = predict(q)
        Jr = (I - (d / r)[:, :, None, None] * (I - axis[:, :, :, None] * axis[:, :, None, :])) * env[:, :, None, None]
        Jt = axis * (-u * v * np.sin(theta) / d)[:, :, None]
        blocks = np.zeros((n, 2, 3, 5))
        blocks[:, :, :, :3] = Jr
        for s in range(2):
            blocks[:, s, :, 3 + s] = Jt[:, s]
        blocks *= cw[:, :, None, None]
        rr = np.broadcast_to(np.arange(n * 6).reshape(n, 2, 3, 1), blocks.shape).ravel()
        cc = np.broadcast_to(f[:, None, None, None] * 5 + np.arange(5), blocks.shape).ravel()
        J = coo_matrix((blocks.ravel(), (rr, cc)), shape=(n * 6, n * 5)).tocsr()
        rhs = Ab - H @ q.ravel() - J.T @ (err * cw[:, :, None]).ravel()
        normal = (H + J.T @ J + diags(np.full(n * 5, 1e-05))).tocsc()
        lo = np.c_[-cfg.get('root_limit', 0.015) * np.tile(rf[:, None], (1, 3)), np.maximum(-theta0 + 1e-05, -np.radians(15) * env)]
        hi = np.c_[cfg.get('root_limit', 0.015) * np.tile(rf[:, None], (1, 3)), np.minimum(np.pi - 0.001 - theta0, np.radians(15) * env)]
        lo = np.minimum(lo, hi)
        low = (lo - q).ravel()
        high = (hi - q).ravel()
        delta = spsolve(normal, rhs)
        fixed = np.zeros(n * 5, bool)
        for _ in range(4):
            hits = (delta < low) | (delta > high)
            if not np.any(hits & ~fixed):
                break
            fixed |= hits
            delta[fixed] = np.clip(delta[fixed], low[fixed], high[fixed])
            free = ~fixed
            delta[free] = spsolve(normal[free][:, free], rhs[free] - normal[free][:, fixed] @ delta[fixed])
        delta = delta.reshape(n, 5)
        before = cost(q)
        alpha = 1.0
        for trial in range(8):
            cand = q + alpha * delta
            cand[:, :3] = np.clip(cand[:, :3], -cfg.get('root_limit', 0.015) * rf[:, None], cfg.get('root_limit', 0.015) * rf[:, None])
            cand[:, 3:] = np.clip(cand[:, 3:], np.maximum(-theta0 + 1e-05, -np.radians(15) * env), np.minimum(np.pi - 0.001 - theta0, np.radians(15) * env))
            after = cost(cand)
            if after <= before:
                break
            alpha *= 0.5
        if after > before:
            delta = (rhs / np.maximum(normal.diagonal(), 1e-10)).reshape(n, 5)
            alpha = 1.0
            for trial in range(12):
                cand = np.clip(q + alpha * delta, lo, hi)
                after = cost(cand)
                if after <= before:
                    break
                alpha *= 0.5
            if after > before:
                break
        q = cand
        costs.append(after)
        if np.max(abs(alpha * delta)) < 1e-06:
            break
    (err, axis, r, d, theta) = predict(q)
    world = old['world'].copy()
    root = root0 + q[:, :3]
    for (s, (a, bone, c)) in enumerate(m.CHAINS[:2]):
        oldaxis = unit(p[:, c] - p[:, a])
        observed = p[:, bone] - p[:, a] - oldaxis * np.sum((p[:, bone] - p[:, a]) * oldaxis, axis=1)[:, None]
        pole = stable_pole(old, seed, s, env[:, s])
        pole = np.einsum('tij,tj->ti', m.align_many(oldaxis, axis[:, s]), pole)
        along = (u[s] ** 2 - v[s] ** 2 + d[:, s] ** 2) / (2 * d[:, s])
        height = u[s] * v[s] * np.sin(theta[:, s]) / d[:, s]
        hip = root + hipoff[:, s]
        knee = hip + axis[:, s] * along[:, None] + pole * height[:, None]
        end = hip + axis[:, s] * d[:, s, None]
        world[:, a] = m.align_many(p[:, bone] - p[:, a], knee - hip) @ old['world'][:, a]
        world[:, bone] = m.align_many(p[:, c] - p[:, bone], end - knee) @ old['world'][:, bone]
        world[~enabled[:, s], a] = old['world'][~enabled[:, s], a]
        world[~enabled[:, s], bone] = old['world'][~enabled[:, s], bone]
    local = m.local_from_world(world)
    (points, world) = m.fk(local, rest, root)
    result = dict(old, root=root, world=world, local=local, points=points)
    return (result, dict(seconds=time.perf_counter() - start, iterations=len(costs), costs=costs, root_added_max_mm=float(np.linalg.norm(q[:, :3], axis=-1).max() * 1000), ankle_change_contact_max_mm=float(np.linalg.norm(points[:, [7, 8]] - goals, axis=-1)[active].max(initial=0) * 1000), bend_added_max_deg=float(np.degrees(abs(q[:, 3:])).max())))

def metrics(data, seed, old, active):
    f = np.degrees(flex(data['points']))
    extra = f - np.degrees(flex(seed['points']))
    mask = active[:-2, :2] & active[1:-1, :2] & active[2:, :2]
    d = np.diff(f, n=2, axis=0)
    de = np.diff(extra, n=2, axis=0)
    bounds = []
    for s in range(2):
        for i in np.flatnonzero(np.diff(active[:, s].astype(int))) + 1:
            bounds.extend(abs(d[max(0, i - 3):min(len(d), i + 3), s]))
    return dict(rms=float(np.sqrt(np.mean(d[mask] ** 2))), added_rms=float(np.sqrt(np.mean(de[mask] ** 2))), boundary_p95=float(np.percentile(bounds, 95)) if bounds else 0, boundary_max=float(max(bounds, default=0)), root_accel_rms=float(np.sqrt(np.mean(np.diff(data['root'], n=2, axis=0) ** 2)) * 1000))
