"""
Free-form SAS Inversion Script for Simulated Ellipsoid Data

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import numpy as np
from ffsi.models import Ellipsoid2D
from ffsi.optimize_galahad import optimize
from ffsi.sensitivity_analysis import sensitivity
from ffsi.utils import contract_tensor

# for simulating distributions
from ffsi.crazy_distributions import crazy_distribution

# for plotting
from ffsi.plotting import *

## Step 0: Discretisation parameters
print("Step 0: Discretise parameters\n")

# contrast
drho = 1

# qx, qy discretisation
nqx = 30
nqy = 30
q_side = np.logspace(-2, 0, 15) # log scale on each side
qx = np.hstack((-q_side[::-1], q_side))
qy = qx.copy()

# rp discretisation
rpl = 50
rpu = 90
nrp = 10

# re discretisation
rel = 200
reu = 600
nre = 9

# theta discretisation
thetal = 20
thetau = 75
ntheta = 8

# phi discretisation
phil = 150
phiu = 240
nphi = 7

# discretise rp, re, theta, phi
rp = np.linspace(rpl, rpu, nrp)
re = np.linspace(rel, reu, nre)
theta = np.linspace(thetal, thetau, ntheta)
phi = np.linspace(phil, phiu, nphi)

print('qx: %d' % nqx)
print('qy: %d' % nqy)
print('rp: linspace(%d,%d,%d)' % (rpl, rpu, nrp))
print('re: linspace(%d,%d,%d)' % (rel, reu, nre))
print('theta: linspace(%d,%d,%d)' % (thetal, thetau, ntheta))
print('phi: linspace(%d,%d,%d)' % (phil, phiu, nphi))

## Step 1: Generate ground truth
print("\nStep 1: Generate ground truth\n")

# generate ground truth distributions
w_rp_true = crazy_distribution(rp, [(1, 60, 3), (2, 70, 4), (2, 80, 3)], 0, 1, 1)
w_re_true = crazy_distribution(re, [(1.5, 300, 20), (1, 400, 20), (2, 500, 20)], 0, 1, 1)
w_theta_true = crazy_distribution(theta, [(4, 30, 5), (2, 50, 5), (2, 65, 5)], 0, 1, 1)
w_phi_true = crazy_distribution(phi, [(2, 170, 10), (2, 200, 10), (4, 220, 10)], 0, 1, 1)

# convert degrees to radians
theta = np.deg2rad(theta)
phi = np.deg2rad(phi)

# plot "true" distributions
plot_ellipsoid_distributions([rp,re,theta,phi], [w_rp_true,w_re_true,w_theta_true,w_phi_true], title='True distributions')

# ground truth of scale and background
scale_true = 0.15
b_true = 2.2e-4
print('b_true: %.2e' % b_true)

# instantiate ellipsoid
sasmodel = Ellipsoid2D()

# compute the ground truth of xi
w_true_list = [w_rp_true, w_re_true]
v_param_list = [rp, re]
V_ave = sasmodel.compute_average_volume(v_param_list, w_true_list)
xi_true = 1e-4 * scale_true / V_ave
print('xi_true: %.2e' % xi_true)

# Compute true G
print('\nComputing full G tensor...')
q_list = [qx,qy]
param_list = [rp, re, theta, phi]
G = sasmodel.compute_scattering_intensity(q_list, param_list, drho)
print('done.')

# compute Gw_true for simulating the intensities
print('\nComputing Gw for simulating the intensities...', end='')
w_true_list = [w_rp_true, w_re_true, w_theta_true, w_phi_true]
Gw_true = contract_tensor(G, w_true_list, skip_axes=[0,1])
print('done.')

# compute model intensities as the intensity data
I_data = xi_true * Gw_true + b_true

# plot intensities
plot_2d_intensity(qx, qy, I_data)

## Step 2: SAS Inversion with true G
print("\nStep 2: SAS inversion with true G\n")
xi_opt, b_opt, w_opt_list = optimize(G, I_data, I_data, sigma=None)

# plot optimized distributions
plot_ellipsoid_distributions([rp,re,theta,phi], w_opt_list, title='Optimized distributions')

# compute Gw_opt for the optimized intensities
print('\nComputing Gw for the optimized intensities...', end='')
Gw_opt = contract_tensor(G, w_opt_list, skip_axes=[0,1])
print('done.')

# compute model intensities as the intensity data
I_opt = xi_opt * Gw_opt + b_opt

# plot optimized intensities
plot_2d_optimized_intensity(qx, qy, I_data, I_opt)

# plot intensity misfit
plot_2d_intensity_misfit(qx, qy, I_data, I_opt)

## Step 3: Sensitivity Analysis
print("\nStep 3: Sensitivity for the optimized parameters\n")
sens_xi, sens_b, sens_w_list, std_xi, std_b, std_w_list = sensitivity(G, I_data, I_data, xi_opt, b_opt, w_opt_list)
