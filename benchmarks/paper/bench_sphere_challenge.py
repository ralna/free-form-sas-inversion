"""
Paper benchmark: spheres with a drastically varying size distribution
(parameters identical to ffsi/scripts/sas_inversion_sphere_challenge.py)

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import cupy as cp
from ffsi.models.sphere import Sphere
from ffsi.utils import contract_tensor, scale_to_xi
from ffsi.crazy_distributions import crazy_distribution
from common import timed, save_problem, run_inversion, save_result

NAME = 'sphere_challenge'
SIGMA = 0.25
RESOLUTIONS = [100, 200, 300, 400]

drho = 25
rl, ru, nr_true = 500, 1000, 300
q = 10 ** cp.linspace(-3, 0, 4000)
r = cp.linspace(rl, ru, nr_true)

# ground truth (generated once here with CuPy's RNG and shared with FFSAS)
w_r_true = crazy_distribution(r, [(4, 580, 10), (6, 630, 20), (10, 700, 20),
                                  (12, 750, 20), (8, 850, 15), (5, 930, 15)],
                              10, 10, 40)
scale_true = 2.5
b_true = 0.14
xi_true = float(scale_to_xi(scale_true, Sphere.compute_average_volume([r], [w_r_true])))

G = Sphere.compute_scattering_intensity([q], [r], drho)
I_data = xi_true * contract_tensor(G, [w_r_true], skip_axes=[0]) + b_true
I_data_std = I_data.copy()  # noise free, relative error bars

grids = {f'r_{n}': cp.linspace(rl, ru, n) for n in RESOLUTIONS}
save_problem(NAME, q=q, r=r, w_r_true=w_r_true, xi_true=xi_true, b_true=b_true,
             drho=drho, I_data=I_data, I_data_std=I_data_std, **grids)

for n in RESOLUTIONS:
    G_n, t_G = timed(cp, Sphere.compute_scattering_intensity, [q], [grids[f'r_{n}']], drho)
    for sigma, tag in ((None, 'noreg'), (SIGMA, 'reg')):
        res = run_inversion(G_n, I_data, I_data_std, sigma, q_axes=[0])
        save_result(f'{NAME}_{n}_gpu_{tag}', res, G_time=t_G, sigma=sigma,
                    device='gpu', nr=n, xi_true=xi_true, b_true=b_true)
