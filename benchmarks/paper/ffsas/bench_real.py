"""
Part of the ffsi paper benchmarks: runs FFSAS (https://github.com/stfc-sciml/ffsas)
on the problems saved by the ffsi drivers in benchmarks/paper. Derived from the
FFSAS example notebooks, Copyright 2021 United Kingdom Research and Innovation,
used under the BSD-3-Clause licence. Run with a Python environment in which FFSAS
is installed.

FFSAS runs of examples 04 (SANS) and 05 (SAXS) on the same data, truncation and
radius grids as ffsi. Converted from examples/04_sphere_SANS/sphere_SANS.ipynb
(maxiter=1000) and examples/05_sphere_SAXS/sphere_SAXS.ipynb (maxiter=500),
both with trust_options={'xtol': 0, 'gtol': 0}.
"""
from common import *
from ffsas.models import Sphere

for dataset, maxiter in (('SANS', 1000), ('SAXS', 500)):
    P = load_problem(f'sphere_{dataset}')
    for device in ('cuda', 'cpu'):
        G, t_G = timed(device, Sphere.compute_G_mini_batch, [P['q']], {'r': P['r']},
                       {'drho': float(P['drho'])}, device=device, log_screen=False)
        g_sys = SASGreensSystem(G, Sphere.get_par_keys_G(), device=device, log_screen=False)
        res = run_inversion(g_sys, device, P['I_data'], P['I_data_std'], maxiter=maxiter,
                            trust_options={'xtol': 0., 'gtol': 0.}, repeats=1)
        save_result(f'sphere_{dataset}_{device}', res, G_time=t_G, device=device,
                    maxiter=maxiter)
