"""
Free-form SAS Inversion Script for Simulated Ellipsoid Data (1D)

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import numpy as np
from ffsi.models.ellipsoid import Ellipsoid
from ffsi.optimize_galahad import optimize
from ffsi.sensitivity_analysis import sensitivity
from ffsi.utils import contract_tensor, scale_to_xi

# for plotting
from ffsi.plotting import *

## Step 0: Discretisation parameters
print("Step 0: Discretise parameters\n")

# contrast
drho = 1

# q discretisation (log)
ql = -3
qu = 0
nq = 200

# rp discretisation
rpl = 5
rpu = 35
nrp = 100

# re discretisation
rel = 200
reu = 600
nre = 99

# discretise q, l, r
q = np.logspace(ql, qu, nq)
rp = np.linspace(rpl, rpu, nrp)
re = np.linspace(rel, reu, nre)

print('q: logspace(%d,%d,%d)' % (ql, qu, nq))
print('rp: linspace(%d,%d,%d)' % (rpl, rpu, nrp))
print('re: linspace(%d,%d,%d)' % (rel, reu, nre))

## Step 1: Generate ground truth
print("\nStep 1: Generate ground truth\n")

# generate ground truth distribution for polar radius
gaussian = np.exp(-(rp - 20)**2 / (2*2**2))
w_rp_true = gaussian
w_rp_true /= w_rp_true.sum()  # normalize

# generate ground truth distribution for equatorial radius
boltzmann = 0.7 * np.exp(-np.abs(re - 400) / 25)
w_re_true = boltzmann
w_re_true /= w_re_true.sum()  # normalize

# plot "true" distributions
plot_ellipsoid_distributions([rp,re], [w_rp_true,w_re_true], title='True distributions')

# ground truth of scale and background
scale_true = 0.15
b_true = 2.2e-4
print('b_true: %.2e' % b_true)

# instantiate ellipsoid
sasmodel = Ellipsoid()

# compute the ground truth of xi
w_true_list = [w_rp_true, w_re_true]
v_param_list = [rp, re]
V_ave = sasmodel.compute_average_volume(v_param_list, w_true_list)
xi_true = scale_to_xi(scale_true, V_ave)
print('xi_true: %.2e' % xi_true)

# Compute true G
print('\nComputing full G tensor...')
q_list = [q]
param_list = [rp, re]
G = sasmodel.compute_scattering_intensity(q_list, param_list, drho)
print('done.')

# compute Gw_true for simulating the intensities
print('\nComputing Gw for simulating the intensities...', end='')
w_true_list = [w_rp_true, w_re_true]
Gw_true = contract_tensor(G, w_true_list, skip_axes=[0])
print('done.')

# compute model intensities as the intensity data
I_data = xi_true * Gw_true + b_true

# plot intensities
plot_1d_intensity(q, I_data)

## Step 2: SAS Inversion with true G
print("\nStep 2: SAS inversion with true G\n")
xi_opt, b_opt, w_opt_list = optimize(G, I_data, I_data, sigma=0.25)

# plot optimized distributions
plot_ellipsoid_distributions([rp,re], w_opt_list, title='Optimized distributions')

# compute Gw_opt for the optimized intensities
print('\nComputing Gw for the optimized intensities...', end='')
Gw_opt = contract_tensor(G, w_opt_list, skip_axes=[0])
print('done.')

# compute model intensities as the intensity data
I_opt = xi_opt * Gw_opt + b_opt

# plot optimized intensities
plot_1d_optimized_intensity(q, I_data, I_opt)

## Step 3: Sensitivity Analysis
print("\nStep 3: Sensitivity for the optimized parameters\n")
sens_xi, sens_b, sens_w_list, std_xi, std_b, std_w_list = sensitivity(G, I_data, I_data, xi_opt, b_opt, w_opt_list)
