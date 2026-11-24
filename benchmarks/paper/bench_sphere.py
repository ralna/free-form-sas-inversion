"""
Paper benchmark: spheres with an analytical bimodal size distribution
(parameters identical to ffsi/scripts/sas_inversion_sphere.py)

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import numpy as np
import cupy as cp
from ffsi.models.sphere import Sphere
from ffsi.utils import contract_tensor, scale_to_xi
from common import timed, save_problem, run_inversion, save_result

NAME = 'sphere_benchmark'
SIGMA = 0.25

# discretisation (as sas_inversion_sphere.py)
drho = 1
q = np.logspace(-3, 0, 200)
r = np.linspace(400, 800, 500)

# ground truth (generated once here and shared with FFSAS)
gaussian = np.exp(-(r - 500)**2 / (2*10**2))
boltzmann = 0.7 * np.exp(-np.abs(r - 700) / 20)
w_r_true = gaussian + boltzmann
w_r_true /= w_r_true.sum()
scale_true = 2
b_true = 0.5
xi_true = scale_to_xi(scale_true, Sphere.compute_average_volume([r], [w_r_true]))

G = Sphere.compute_scattering_intensity([q], [r], drho)
I_data = xi_true * contract_tensor(G, [w_r_true], skip_axes=[0]) + b_true
np.random.seed(0)
I_data_std = (np.random.rand(len(q)) * 0.1 + 0.2) * I_data

save_problem(NAME, q=q, r=r, w_r_true=w_r_true, xi_true=xi_true, b_true=b_true,
             drho=drho, I_data=I_data, I_data_std=I_data_std)

for device, xp in (('gpu', cp), ('cpu', np)):
    q_d, r_d, I_d, S_d = (xp.asarray(a) for a in (q, r, I_data, I_data_std))
    G_d, t_G = timed(xp, Sphere.compute_scattering_intensity, [q_d], [r_d], drho)
    for sigma, tag in ((SIGMA, 'reg'), (None, 'noreg')):
        res = run_inversion(G_d, I_d, S_d, sigma, q_axes=[0])
        save_result(f'{NAME}_{device}_{tag}', res, G_time=t_G, sigma=sigma,
                    device=device, xi_true=float(xi_true), b_true=b_true)
