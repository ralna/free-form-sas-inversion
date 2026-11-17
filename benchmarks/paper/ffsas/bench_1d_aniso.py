"""
Part of the ffsi paper benchmarks: runs FFSAS (https://github.com/stfc-sciml/ffsas)
on the problems saved by the ffsi drivers in benchmarks/paper. Derived from the
FFSAS example notebooks, Copyright 2021 United Kingdom Research and Innovation,
used under the BSD-3-Clause licence. Run with a Python environment in which FFSAS
is installed.

FFSAS runs for orientation-averaged (1D) cylinders and spheroids. FFSAS has no
such models, so the ffsi Green tensor saved with the problem is passed to the
model-independent SASGreensSystem. Settings follow the FFSAS sphere benchmark
notebook (maxiter=200, xtol=gtol=0).
"""
from common import *

TRUST = {'xtol': 0., 'gtol': 0.}
MAXITER = 200

for name, keys in (('cylinder1d', ['l', 'r']), ('ellipsoid1d', ['rp', 're'])):
    P = load_problem(name)
    for device in ('cuda', 'cpu'):
        g_sys = SASGreensSystem(P['G'], keys, device=device, log_screen=False)
        check_data(g_sys, {k: P[f'w_{k}_true'] for k in keys}, float(P['xi_true']),
                   float(P['b_true']), P['I_data'])
        res = run_inversion(g_sys, device, P['I_data'], P['I_data_std'], maxiter=MAXITER,
                            trust_options=TRUST, repeats=3)
        save_result(f'{name}_{device}', res, G_time=float('nan'), device=device,
                    maxiter=MAXITER)
