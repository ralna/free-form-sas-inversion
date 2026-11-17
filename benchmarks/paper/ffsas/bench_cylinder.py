"""
Part of the ffsi paper benchmarks: runs FFSAS (https://github.com/stfc-sciml/ffsas)
on the problems saved by the ffsi drivers in benchmarks/paper. Derived from the
FFSAS example notebooks, Copyright 2021 United Kingdom Research and Innovation,
used under the BSD-3-Clause licence. Run with a Python environment in which FFSAS
is installed.

FFSAS run of example 03 (2D cylinders, four polydisperse parameters) on the
shared ffsi problem. Converted from examples/03_cylinder/cylinder.ipynb
(upstream settings: maxiter=200, batch_size=8, trust_options={'xtol': 0.}).
Usage: python bench_cylinder.py [cylinder2d|ellipsoid2d]
"""
import sys
from common import *
from ffsas.models import Cylinder, Ellipsoid

NAME = sys.argv[1] if len(sys.argv) > 1 else 'cylinder2d'
MODEL, KEYS = {'cylinder2d': (Cylinder, ['l', 'r', 'theta', 'phi']),
               'ellipsoid2d': (Ellipsoid, ['rp', 're', 'theta', 'phi'])}[NAME]
MAXITER = 200
device = 'cuda'
P = load_problem(NAME)

par_dict = {k: P[k] for k in KEYS}
G, t_G = timed(device, MODEL.compute_G_mini_batch, [P['qx'], P['qy']], par_dict,
               {'drho': float(P['drho'])}, batch_size=8, device=device, log_screen=False)
g_sys = SASGreensSystem(G, MODEL.get_par_keys_G(), batch_size=8, device=device, log_screen=False)
w_true = {k: P[f'w_{k}_true'] for k in KEYS}
check_data(g_sys, w_true, float(P['xi_true']), float(P['b_true']), P['I_data'])
res = run_inversion(g_sys, device, P['I_data'], P['I_data_std'], maxiter=MAXITER,
                    trust_options={'xtol': 0.}, repeats=1)
save_result(f'{NAME}_{device}', res, G_time=t_G, device=device, maxiter=MAXITER)
