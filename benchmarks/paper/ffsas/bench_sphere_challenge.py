"""
Part of the ffsi paper benchmarks: runs FFSAS (https://github.com/stfc-sciml/ffsas)
on the problems saved by the ffsi drivers in benchmarks/paper. Derived from the
FFSAS example notebooks, Copyright 2021 United Kingdom Research and Innovation,
used under the BSD-3-Clause licence. Run with a Python environment in which FFSAS
is installed.

FFSAS run of example 02 (sphere challenge) on the shared ffsi problem.
Converted from examples/02_sphere_challenge/sphere_challenge.ipynb
(maxiter=1000, auto_scaling=True, default trust-constr tolerances).
"""
from common import *
from ffsas.models import Sphere

NAME = 'sphere_challenge'
device = 'cuda'
P = load_problem(NAME)

# check the forward model on the ground-truth grid
G = Sphere.compute_G_mini_batch([P['q']], {'r': P['r']}, {'drho': float(P['drho'])},
                                device=device, log_screen=False)
g_sys = SASGreensSystem(G, Sphere.get_par_keys_G(), device=device, log_screen=False)
check_data(g_sys, {'r': P['w_r_true']}, float(P['xi_true']), float(P['b_true']), P['I_data'])

for n in [100, 200, 300, 400]:
    G, t_G = timed(device, Sphere.compute_G_mini_batch, [P['q']], {'r': P[f'r_{n}']},
                   {'drho': float(P['drho'])}, device=device, log_screen=False)
    g_sys = SASGreensSystem(G, Sphere.get_par_keys_G(), device=device, log_screen=False)
    res = run_inversion(g_sys, device, P['I_data'], P['I_data_std'], maxiter=1000,
                        trust_options={}, repeats=1)
    save_result(f'{NAME}_{n}_{device}', res, G_time=t_G, device=device, nr=n, maxiter=1000)
