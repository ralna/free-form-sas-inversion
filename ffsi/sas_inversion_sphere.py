"""
Free-form SAS Inversion Script for Simulated Sphere Data

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import numpy as np
from ffsi.models import Sphere
from ffsi.optimize_galahad import optimize
from ffsi.utils import contract_tensor

# for plotting
import matplotlib.pyplot as plt

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
plt.figure()
plt.plot(r, w_r_true * 100) # x100 to percent
plt.grid()
plt.title("Ground truth distributions")
plt.xlabel(r"Radius $r$ ($\AA$)")
plt.ylabel(r"Weights $w$ (%)")
plt.show()

# ground truth of scale and background
scale_true = 2
b_true = 0.5
print('b_true: %.2e' % b_true)

# compute the ground truth of xi
w_true_dict = {'r':w_r_true}
v_param_dict = {'r':r}
V_ave = Sphere.compute_average_V(v_param_dict, w_true_dict)
xi_true = 1e-4 * scale_true / V_ave
print('xi_true: %.2e' % xi_true)

# Compute true G
print('\nComputing full G tensor...')
q_list = [q]
param_dict = {'r':r}
const_dict = {'drho':drho}
G = Sphere.compute_G(q_list, param_dict, const_dict)
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
plt.figure()
plt.errorbar(q, I_data, yerr=I_data_std, ecolor='gray')
plt.grid()
plt.xscale('log')
plt.yscale('log')
plt.title('Intensity')
plt.xlabel(r"Scattering vector $q$ ($\AA^{-1}$)")
plt.ylabel(r"Intensity $I$ ($\mathrm{cm}^{-1}$)")
plt.show()

## Step 2: SAS Inversion with true G
print("\nStep 2: SAS inversion with true G\n")
xi_opt, b_opt, w_opt_list = optimize(G, I_data, I_data_std, sigma=0.25)

# plot optimized distributions
plt.figure()
plt.plot(r, w_opt_list[0] * 100) # x100 to percent
plt.grid()
plt.title("Optimized distributions")
plt.xlabel(r"Radius $r$ ($\AA$)")
plt.ylabel(r"Weights $w$ (%)")
plt.show()

# compute Gw_opt for the optimized intensities
print('\nComputing Gw for the optimized intensities...', end='')
Gw_opt = contract_tensor(G, w_opt_list, skip_axes=[0])
print('done.')

# compute model intensities as the intensity data
I_opt = xi_opt * Gw_opt + b_opt

# plot optimized intensities
plt.figure()
plt.grid()
plt.errorbar(q, I_data, yerr=I_data_std, ecolor='gray', marker='o', markerfacecolor='none')
plt.plot(q, I_opt, color='red', zorder=5)
plt.xscale('log')
plt.yscale('log')
plt.title('Optimized Intensity')
plt.xlabel(r"Scattering vector $q$ ($\AA^{-1}$)")
plt.ylabel(r"Intensity $I$ ($\mathrm{cm}^{-1}$)")
plt.legend(['Fit','Data'])
plt.show()
