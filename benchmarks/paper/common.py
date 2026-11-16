"""
Shared helpers for the paper benchmark drivers (timing, problem/result I/O).

Output goes to $FFSI_PAPER_DIR (default: benchmarks/paper/output).

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import json
import os
import time
from pathlib import Path

import numpy as np

from ffsi.array_module import get_array_module
from ffsi.optimize_galahad import optimize
from ffsi.utils import contract_tensor

# output directory for problems, results and figures used in the paper
REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = Path(os.environ.get('FFSI_PAPER_DIR', Path(__file__).resolve().parent / 'output'))
PROBLEM_DIR = PAPER_DIR / 'results' / 'problems'
RESULT_DIR = PAPER_DIR / 'results' / 'ffsi'


def to_numpy(a):
    """Return a NumPy copy of a NumPy/CuPy array (or scalar)."""
    if hasattr(a, 'get'):
        return a.get()
    return np.asarray(a)


def sync(xp):
    """Block until all queued GPU work has finished (no-op on the CPU)."""
    if xp.__name__ == 'cupy':
        xp.cuda.Device().synchronize()


def timed(xp, func, *args, **kwargs):
    """Run func(*args, **kwargs) and return (result, wall-clock seconds)."""
    sync(xp)
    t0 = time.perf_counter()
    out = func(*args, **kwargs)
    sync(xp)
    return out, time.perf_counter() - t0


def save_problem(name, **arrays):
    """Save the problem definition (shared with the FFSAS scripts) to an npz file."""
    PROBLEM_DIR.mkdir(parents=True, exist_ok=True)
    path = PROBLEM_DIR / f'{name}.npz'
    np.savez(path, **{k: to_numpy(v) for k, v in arrays.items()})
    print(f'Saved problem to {path}')


def chi2(G, w_list, xi, b, I_data, I_data_std, q_axes):
    """Unregularized chi^2 = sum(((xi G w + b) - mu) / sigma)^2."""
    xp = get_array_module(G)
    w_list = [xp.asarray(w) for w in w_list]
    I_opt = xi * contract_tensor(G, w_list, skip_axes=q_axes) + b
    eps = (I_opt - I_data) / I_data_std
    return float(xp.sum(eps ** 2)), to_numpy(I_opt)


def run_inversion(G, I_data, I_data_std, sigma, q_axes, repeats=3, warmup=True):
    """
    Run the GALAHAD inversion `repeats` times (after an optional warm-up run)
    and return the result of the last run with timings and convergence history.
    """
    xp = get_array_module(G)
    if warmup:
        optimize(G, I_data, I_data_std, sigma=sigma)
    times = []
    for _ in range(repeats):
        history = []
        sync(xp)
        t0 = time.perf_counter()
        xi, b, w_list = optimize(G, I_data, I_data_std, sigma=sigma, history=history)
        sync(xp)
        times.append(time.perf_counter() - t0)
    c2, I_opt = chi2(G, w_list, xi, b, I_data, I_data_std, q_axes)
    hist = np.array([(t - t0, c) for t, c in history])
    return dict(xi=float(xi), b=float(b), w_list=[to_numpy(w) for w in w_list],
                chi2=c2, I_opt=I_opt, solve_times=times,
                solve_time=float(np.median(times)), history=hist,
                n_evals=len(history))


def save_result(name, res, **extra):
    """Save a result as npz (arrays) plus json (scalars) for the paper."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {f'w{i}': w for i, w in enumerate(res['w_list'])}
    arrays.update(I_opt=res['I_opt'], history=res['history'])
    np.savez(RESULT_DIR / f'{name}.npz', **arrays)
    summary = {k: v for k, v in res.items() if k not in ('w_list', 'I_opt', 'history')}
    summary.update(extra)
    with open(RESULT_DIR / f'{name}.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f'\n[{name}] chi2 = {res["chi2"]:.6e}, solve time (median of '
          f'{len(res["solve_times"])}) = {res["solve_time"]:.3f} s')
