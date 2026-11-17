"""
Paper benchmark: orientation-averaged (1D) cylinders and ellipsoids with two
polydisperse size parameters (parameters identical to
ffsi/scripts/sas_inversion_cylinder.py and sas_inversion_ellipsoid.py).

FFSAS has no orientation-averaged models, so the Green tensor G is saved with
the problem and passed unchanged to FFSAS's (model-independent) solver.

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import numpy as np
import cupy as cp
from ffsi.models.cylinder import Cylinder
from ffsi.models.ellipsoid import Ellipsoid
from ffsi.utils import contract_tensor, scale_to_xi
from common import timed, save_problem, run_inversion, save_result

drho = 1
q = np.logspace(-3, 0, 200)

# size grids and ground truths as in the corresponding ffsi scripts
# rod-like cylinders: Gaussian length and radius distributions
lengths = np.linspace(200, 600, 50)
radii = np.linspace(5, 35, 50)
w_long = np.exp(-(lengths - 400)**2 / (2*30**2))
w_long /= w_long.sum()
w_rad = np.exp(-(radii - 20)**2 / (2*2**2))
w_rad /= w_rad.sum()

# oblate ellipsoids: Gaussian polar and equatorial radius distributions
rp = np.linspace(20, 60, 50)
re = np.linspace(200, 600, 50)
w_rp = np.exp(-(rp - 40)**2 / (2*4**2))
w_rp /= w_rp.sum()
w_re = np.exp(-(re - 400)**2 / (2*30**2))
w_re /= w_re.sum()

CASES = {
    'cylinder1d': (Cylinder, ['l', 'r'], [lengths, radii], [w_long, w_rad]),
    'ellipsoid1d': (Ellipsoid, ['rp', 're'], [rp, re], [w_rp, w_re]),
}

for name, (model, keys, grids, w_true) in CASES.items():
    scale_true = 0.15
    b_true = 2.2e-4
    xi_true = float(scale_to_xi(scale_true, model.compute_average_volume(grids, w_true)))
    G = model.compute_scattering_intensity([q], grids, drho)
    I_data = xi_true * contract_tensor(G, w_true, skip_axes=[0]) + b_true
    I_data_std = I_data.copy()  # noise free, relative error bars

    save_problem(name, q=q, G=G, xi_true=xi_true, b_true=b_true, drho=drho,
                 I_data=I_data, I_data_std=I_data_std,
                 **{k: g for k, g in zip(keys, grids)},
                 **{f'w_{k}_true': w for k, w in zip(keys, w_true)})

    for device, xp in (('gpu', cp), ('cpu', np)):
        q_d, I_d, S_d = (xp.asarray(a) for a in (q, I_data, I_data_std))
        grids_d = [xp.asarray(g) for g in grids]
        G_d, t_G = timed(xp, model.compute_scattering_intensity, [q_d], grids_d, drho)
        # noise-free exact-recovery tests: no smoothing
        res = run_inversion(G_d, I_d, S_d, None, q_axes=[0])
        save_result(f'{name}_{device}_noreg', res, G_time=t_G, sigma=None,
                    device=device, xi_true=xi_true, b_true=b_true)
