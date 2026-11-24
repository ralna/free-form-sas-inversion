"""
Paper benchmark: 2D cylinders with four polydisperse parameters
(parameters identical to ffsi/scripts/sas_inversion_cylinder2d_cupy.py)

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import cupy as cp
from ffsi.models.cylinder2d import Cylinder2D
from ffsi.utils import contract_tensor, scale_to_xi
from ffsi.crazy_distributions import crazy_distribution
from common import timed, save_problem, run_inversion, save_result

NAME = 'cylinder2d'

drho = 1
qx = cp.linspace(-0.75, 0.75, 60)
qy = cp.linspace(-0.75, 0.75, 60)
l = cp.linspace(200, 600, 18)
r = cp.linspace(50, 90, 17)
theta = cp.linspace(5, 60, 16)
phi = cp.linspace(150, 240, 15)

# ground truth (generated once here with CuPy's RNG and shared with FFSAS)
w_l_true = crazy_distribution(l, [(1.5, 300, 20), (1, 400, 20), (2, 500, 20)], 1, 1, 1)
w_r_true = crazy_distribution(r, [(1, 60, 3), (2, 70, 4), (2, 80, 3)], 1, 1, 1)
w_theta_true = crazy_distribution(theta, [(4, 15, 5), (2, 35, 5), (2, 50, 5)], 2, 1, 1)
w_phi_true = crazy_distribution(phi, [(2, 170, 10), (2, 200, 10), (4, 220, 10)], 3, 1, 1)
theta = cp.deg2rad(theta)
phi = cp.deg2rad(phi)

scale_true = 0.15
b_true = 2.2e-4
xi_true = float(scale_to_xi(scale_true, Cylinder2D.compute_average_volume([l, r], [w_l_true, w_r_true])))

G, t_G = timed(cp, Cylinder2D.compute_scattering_intensity, [qx, qy], [l, r, theta, phi], drho)
w_true = [w_l_true, w_r_true, w_theta_true, w_phi_true]
I_data = xi_true * contract_tensor(G, w_true, skip_axes=[0, 1]) + b_true

save_problem(NAME, qx=qx, qy=qy, l=l, r=r, theta=theta, phi=phi,
             w_l_true=w_l_true, w_r_true=w_r_true, w_theta_true=w_theta_true,
             w_phi_true=w_phi_true, xi_true=xi_true, b_true=b_true, drho=drho,
             I_data=I_data, I_data_std=I_data)

res = run_inversion(G, I_data, I_data, None, q_axes=[0, 1])
save_result(f'{NAME}_gpu_noreg', res, G_time=t_G, sigma=None, device='gpu',
            xi_true=xi_true, b_true=b_true)
