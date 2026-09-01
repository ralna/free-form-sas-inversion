"""
Free-form SAS Inversion Script for Simulated Cylinder Data (1D)

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import numpy as np
from ffsi.models import Cylinder
from ffsi.optimize_galahad import optimize
from ffsi.sensitivity_analysis import sensitivity
from ffsi.utils import contract_tensor

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

# l discretisation
ll = 200
lu = 600
nl = 100

# r discretisation
rl = 5
ru = 35
nr = 99

# discretise q, l, r
q = np.logspace(ql, qu, nq)
l = np.linspace(ll, lu, nl)
r = np.linspace(rl, ru, nr)

print('q: logspace(%d,%d,%d)' % (ql, qu, nq))
print('l: linspace(%d,%d,%d)' % (ll, lu, nl))
print('r: linspace(%d,%d,%d)' % (rl, ru, nr))

## Step 1: Generate ground truth
print("\nStep 1: Generate ground truth\n")

# generate ground truth distribution for length
boltzmann = 0.7 * np.exp(-np.abs(l - 400) / 25)
w_l_true = boltzmann
w_l_true /= w_l_true.sum()  # normalize

# generate ground truth distribution for radius
gaussian = np.exp(-(r - 20)**2 / (2*2**2))
w_r_true = gaussian
w_r_true /= w_r_true.sum()  # normalize

# plot "true" distributions
plot_cylinder_distributions([l,r], [w_l_true,w_r_true], title='True distributions')

# ground truth of scale and background
scale_true = 0.15
b_true = 2.2e-4
print('b_true: %.2e' % b_true)

# instantiate cylinder
sasmodel = Cylinder()

# compute the ground truth of xi
w_true_list = [w_l_true, w_r_true]
v_param_list = [l, r]
V_ave = sasmodel.compute_average_volume(v_param_list, w_true_list)
xi_true = 1e-4 * scale_true / V_ave
print('xi_true: %.2e' % xi_true)

# Compute true G
print('\nComputing full G tensor...')
q_list = [q]
param_list = [l, r]
G = sasmodel.compute_scattering_intensity(q_list, param_list, drho)
print('done.')

# compute Gw_true for simulating the intensities
print('\nComputing Gw for simulating the intensities...', end='')
w_true_list = [w_l_true, w_r_true]
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
plot_cylinder_distributions([l,r], w_opt_list, title='Optimized distributions')

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
