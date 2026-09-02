"""
Free-form SAS Inversion Script for Simulated Sphere Data
with simple bounds on the scale xi and background b

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import numpy as np
from ffsi.models.sphere import Sphere
from ffsi.optimize_galahad_bounded import optimize_bounded
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

# r discretisation
rl = 400
ru = 800
nr = 500

# discretise q and r
q = np.logspace(ql, qu, nq)
r = np.linspace(rl, ru, nr)

print('q: logspace(%d,%d,%d)' % (ql, qu, nq))
print('r: linspace(%d,%d,%d)' % (rl, ru, nr))

## Step 1: Generate ground truth
print("\nStep 1: Generate ground truth\n")

# ground truth radii distributions
gaussian = np.exp(-(r - 500)**2 / (2*10**2))
boltzmann = 0.7 * np.exp(-np.abs(r - 700) / 20)
w_r_true = gaussian + boltzmann
w_r_true /= w_r_true.sum()  # normalize

# plot ground truth radii distributions
plot_sphere_distribution(r, w_r_true, title='Ground truth distribution')

# ground truth of scale and background
scale_true = 2
b_true = 0.5
print('b_true: %.2e' % b_true)

# instantiate sphere
sasmodel = Sphere()

# compute the ground truth of xi
w_true_list = [w_r_true]
v_param_list = [r]
V_ave = sasmodel.compute_average_volume(v_param_list, w_true_list)
xi_true = scale_to_xi(scale_true, V_ave)
print('xi_true: %.2e' % xi_true)

# put some bounds on xi and b
xi_lb, xi_ub = [0,1]
b_lb, b_ub = [0.2,0.45]

# Compute true G
print('\nComputing full G tensor...')
q_list = [q]
param_list = [r]
G = sasmodel.compute_scattering_intensity(q_list, param_list, drho)
print('done.')

# compute Gw_true for simulating the intensities
print('\nComputing Gw for simulating the intensities...', end='')
w_true_list = [w_r_true]
Gw_true = contract_tensor(G, w_true_list, skip_axes=[0])
print('done.')

# compute model intensities as the intensity data
I_data = xi_true * Gw_true + b_true

# add a 20%~30% error bar
np.random.seed(0)
I_data_std = (np.random.rand(len(q)) * 0.1 + 0.2) * I_data

# plot intensities
plot_1d_intensity(q, I_data, I_data_std)

## Step 2: SAS Inversion with true G
print("\nStep 2: SAS inversion with true G\n")
xi_opt, b_opt, w_opt_list = optimize_bounded(G, I_data, I_data_std, xi_lb, xi_ub, b_lb, b_ub, sigma=0.25)

# plot optimized distributions
plot_sphere_distribution(r, w_opt_list[0], title='Optimized distribution')

# compute Gw_opt for the optimized intensities
print('\nComputing Gw for the optimized intensities...', end='')
Gw_opt = contract_tensor(G, w_opt_list, skip_axes=[0])
print('done.')

# compute model intensities as the intensity data
I_opt = xi_opt * Gw_opt + b_opt

# plot optimized intensities
plot_1d_optimized_intensity(q, I_data, I_opt, I_data_std)
