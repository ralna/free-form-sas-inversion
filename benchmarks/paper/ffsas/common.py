"""
Part of the ffsi paper benchmarks: runs FFSAS (https://github.com/stfc-sciml/ffsas)
on the problems saved by the ffsi drivers in benchmarks/paper. Derived from the
FFSAS example notebooks, Copyright 2021 United Kingdom Research and Innovation,
used under the BSD-3-Clause licence. Run with a Python environment in which FFSAS
is installed.

Shared helpers for running FFSAS on exactly the same problems as ffsi.

The ground truth and data are generated ONCE by the ffsi drivers (CuPy/NumPy
RNG) and loaded here from npz files, so that the different random streams of
PyTorch and CuPy/NumPy cannot make the two codes solve different problems.
"""
import json
import os
import time
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import torch
import ffsas  # noqa: F401  (sets torch default dtype to float64)
from ffsas.system import SASGreensSystem

PAPER_DIR = Path(os.environ.get('FFSI_PAPER_DIR', Path(__file__).resolve().parents[1] / 'output'))
PROBLEM_DIR = PAPER_DIR / 'results' / 'problems'
RESULT_DIR = PAPER_DIR / 'results' / 'ffsas'


def load_problem(name):
    data = np.load(PROBLEM_DIR / f'{name}.npz')
    return {k: torch.tensor(data[k], dtype=torch.float64) for k in data.files}


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def timed(sync_device, func, *args, **kwargs):
    sync(sync_device)
    t0 = time.perf_counter()
    out = func(*args, **kwargs)
    sync(sync_device)
    return out, time.perf_counter() - t0


def check_data(g_sys, w_dict, xi, b, I_data, rtol=1e-10):
    """Check FFSAS's G reproduces the ffsi-generated data (same G, same truth)."""
    I = g_sys.compute_intensity(w_dict, xi, b)
    err = float(torch.max(torch.abs(I - I_data) / torch.abs(I_data)))
    print(f'max relative difference between FFSAS and ffsi intensities: {err:.2e}')
    assert err < rtol, 'FFSAS forward model does not reproduce the shared data'
    return err


def run_inversion(g_sys, device, mu, sigma, maxiter, trust_options, repeats=1, warmup=True):
    """Time solve_inverse (after a 1-iteration warm-up) and record chi^2 history."""
    if warmup:
        g_sys.solve_inverse(mu, sigma, maxiter=1, verbose=0,
                            trust_options=dict(trust_options),
                            returns_intensity_sensitivity_uncertainty=False)
    times = []
    for _ in range(repeats):
        history = []

        def callback(_x, opt_res, history=history):
            history.append((time.perf_counter(), float(opt_res['fun']), int(opt_res['nit'])))
            return False

        g_sys._call_back_save = callback
        sync(device)
        t0 = time.perf_counter()
        res = g_sys.solve_inverse(mu, sigma, maxiter=maxiter, verbose=1,
                                  trust_options=dict(trust_options),
                                  returns_intensity_sensitivity_uncertainty=False)
        sync(device)
        times.append(time.perf_counter() - t0)

    # unregularized chi^2 = sum(((xi G w + b) - mu) / sigma)^2
    I_opt = g_sys.compute_intensity(res['w_dict'], res['xi'], res['b'])
    chi2 = float(torch.sum(((I_opt - mu) / sigma) ** 2))
    # convert scipy's objective history to chi^2 using the final value
    fun_final = float(res['opt_res']['fun'])
    scale = chi2 / fun_final if fun_final > 0 else 1.0
    hist = np.array([(t - t0, f * scale, n) for t, f, n in history])
    return dict(xi=float(res['xi']), b=float(res['b']),
                w_list=[res['w_dict'][k].cpu().numpy() for k in g_sys._par_keys],
                chi2=chi2, I_opt=I_opt.cpu().numpy(), solve_times=times,
                solve_time=float(np.median(times)), history=hist,
                nit=int(res['opt_res']['nit']), wct_scipy=float(res['wct']),
                fun_to_chi2=scale)


def save_result(name, res, **extra):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {f'w{i}': w for i, w in enumerate(res['w_list'])}
    arrays.update(I_opt=res['I_opt'], history=res['history'])
    np.savez(RESULT_DIR / f'{name}.npz', **arrays)
    summary = {k: v for k, v in res.items() if k not in ('w_list', 'I_opt', 'history')}
    summary.update(extra)
    with open(RESULT_DIR / f'{name}.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f'\n[{name}] chi2 = {res["chi2"]:.6e}, solve time = {res["solve_time"]:.3f} s, '
          f'nit = {res["nit"]}')
