"""
Part of the ffsi paper benchmarks: runs FFSAS (https://github.com/stfc-sciml/ffsas)
on the problems saved by the ffsi drivers in benchmarks/paper. Derived from the
FFSAS example notebooks, Copyright 2021 United Kingdom Research and Innovation,
used under the BSD-3-Clause licence. Run with a Python environment in which FFSAS
is installed.

FFSAS run of example 01 (sphere benchmark) on the shared ffsi problem.
Converted from examples/01_sphere_benchmark/sphere_benchmark.ipynb.
"""
from common import *
from ffsas.models import Sphere

NAME = 'sphere_benchmark'
P = load_problem(NAME)
TRUST = {'xtol': 0., 'gtol': 0.}  # as in the notebook: run all maxiter iterations

for device in ('cuda', 'cpu'):
    G, t_G = timed(device, Sphere.compute_G_mini_batch, [P['q']], {'r': P['r']},
                   {'drho': float(P['drho'])}, device=device, log_screen=False)
    g_sys = SASGreensSystem(G, Sphere.get_par_keys_G(), device=device, log_screen=False)
    check_data(g_sys, {'r': P['w_r_true']}, float(P['xi_true']), float(P['b_true']), P['I_data'])
    res = run_inversion(g_sys, device, P['I_data'], P['I_data_std'], maxiter=200,
                        trust_options=TRUST, repeats=3)
    save_result(f'{NAME}_{device}', res, G_time=t_G, device=device, maxiter=200)
