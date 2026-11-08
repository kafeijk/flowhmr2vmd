"""Locate an external Python with numpy+scipy and run the contact solver."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


class SolverError(Exception):
    pass


def _python_candidates(folder):
    env = os.environ.get('RGBMOCAP_PYTHON')
    if env:
        yield Path(env)
    for name in ('python', 'python3'):
        found = shutil.which(name)
        if found:
            yield Path(found)
    yield Path(sys.executable)
    folder = Path(folder)
    base = folder
    for _ in range(0, 8):
        for rel in ('python.exe', 'envs/python.exe', 'Scripts/python.exe',
                    'venv/Scripts/python.exe', '.venv/Scripts/python.exe'):
            path = base / rel
            if path.is_file():
                yield path
        envs = base / 'envs'
        if envs.is_dir():
            for child in envs.iterdir():
                for rel in ('python.exe', 'Scripts/python.exe'):
                    path = child / rel
                    if path.is_file():
                        yield path
        base = base.parent
        if base == base.parent:
            break


def _works(path):
    try:
        result = subprocess.run(
            [str(path), '-c', 'import numpy, scipy; print(1)'],
            capture_output=True, text=True, timeout=25,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        return result.returncode == 0 and '1' in result.stdout
    except Exception:
        return False


def find_python(folder):
    seen = set()
    for path in _python_candidates(folder):
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        if 'windowsapps' in key:
            continue
        if path == Path(sys.executable) and path.name.lower() not in (
                'python.exe', 'python', 'python3'):
            continue
        if path.is_file() and _works(path):
            return path
    return None


def run_solver(request, output, folder):
    python = find_python(folder)
    if python is None:
        raise SolverError(
            '未找到带 numpy/scipy 的 Python 解释器，无法进行脚部接触修正。'
            '请安装 Python 3 并执行 pip install numpy scipy，'
            '或设置环境变量 RGBMOCAP_PYTHON 指向 python.exe。')
    worker = Path(__file__).resolve().parents[1] / 'worker' / 'solve_worker.py'
    command = [str(python), str(worker), '--request', str(request),
               '--output', str(output)]
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=900,
            env=env, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except subprocess.TimeoutExpired as exc:
        raise SolverError('接触解算超时（超过 15 分钟）。') from exc
    if result.returncode != 0:
        error_file = Path(output) / 'error.txt'
        detail = error_file.read_text(encoding='utf8') if error_file.is_file() else ''
        detail = detail or (result.stderr or result.stdout or '')
        raise SolverError('接触解算失败：\n' + detail[-1800:])
    stance = Path(output) / 'stance.npz'
    if not stance.is_file():
        raise SolverError('接触解算没有生成结果文件。')
    score = {}
    score_file = Path(output) / 'score.json'
    if score_file.is_file():
        score = json.loads(score_file.read_text(encoding='utf8'))
    return stance, score


def write_request(path, source, target_rest, target_names, geometry,
                  source_names, lock_toes, project=None):
    payload = {
        'source_points': source['points'].astype(np.float64),
        'source_world': source['world'].astype(np.float64),
        'source_rest': source['rest'].astype(np.float64),
        'target_rest': target_rest.astype(np.float64),
        'active': source['active'].astype(bool),
        'fps': np.array(float(source['fps'])),
        'lock_toes': np.array(bool(lock_toes)),
        'use_project_pt': np.array(bool(project and project['use_project_pt'])),
        'source_mode': np.array('project' if project else 'final'),
        'names_json': np.array(json.dumps(
            dict(source=list(source_names), target=list(target_names)), ensure_ascii=False)),
    }
    if source.get('neutral') is not None:
        payload['source_neutral'] = np.asarray(source['neutral'], np.float64)
    if source.get('raw_file'):
        payload['project_raw_path'] = np.array(str(source['raw_file']))
    if project:
        payload['project_model_path'] = np.array(str(project['model_path']))
        payload['project_pt_path'] = np.array(str(project['pt_path']))
        if project.get('origin') is not None:
            payload['source_origin'] = np.asarray(project['origin'], np.float64)
    for key, value in geometry.items():
        payload[key] = np.asarray(value)
    np.savez(path, **payload)


def read_stance(path):
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in
                ('root', 'world', 'local', 'points', 'rest', 'names', 'ratio')}


def make_temp():
    return Path(tempfile.mkdtemp(prefix='rgbmocap_'))
