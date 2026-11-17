"""
Free-form SAS Inversion Script for Simulated Sphere Data with a
Drastically Varying Size Distribution (multi-resolution challenge)

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import time
import cupy as cp
from ffsi.models.sphere import Sphere
from ffsi.optimize_galahad import optimize
from ffsi.utils import contract_tensor, scale_to_xi

# for simulating distributions
from ffsi.crazy_distributions import crazy_distribution

# for plotting
from ffsi.plotting import *

## Step 0: Discretisation parameters
print("Step 0: Discretise parameters\n")

# contrast
drho = 25

# q discretisation (log)
ql = -3
qu = 0
nq = 4000

# r discretisation of the ground truth
rl = 500
ru = 1000
nr = 300

# discretise q and r
q = cp.logspace(ql, qu, nq)
r = cp.linspace(rl, ru, nr)

print('q: logspace(%d,%d,%d)' % (ql, qu, nq))
print('r: linspace(%d,%d,%d)' % (rl, ru, nr))

## Step 1: Generate ground truth
print("\nStep 1: Generate ground truth\n")

# ground truth radii distribution: six Gaussians plus large-amplitude noise
w_r_true = crazy_distribution(r, [(4, 580, 10), (6, 630, 20), (10, 700, 20),
                                  (12, 750, 20), (8, 850, 15), (5, 930, 15)],
                              10, 10, 40)

# plot ground truth radii distribution
plot_sphere_distribution(r.get(), w_r_true.get(), title='Ground truth distribution')

# ground truth of scale and background
scale_true = 2.5
b_true = 0.14
print('b_true: %.2e' % b_true)

# compute the ground truth of xi
V_ave = Sphere.compute_average_volume([r], [w_r_true])
xi_true = scale_to_xi(scale_true, V_ave)
print('xi_true: %.2e' % xi_true)

# Compute true G
print('\nComputing full G tensor on GPU...')
G = Sphere.compute_scattering_intensity([q], [r], drho)
print('done.')

# compute model intensities as the (noise-free) intensity data
I_data = xi_true * contract_tensor(G, [w_r_true], skip_axes=[0]) + b_true

# plot intensities
plot_1d_intensity(q.get(), I_data.get())

## Step 2: SAS Inversion at several resolutions of r
print("\nStep 2: SAS inversion at several resolutions\n")
for nr_fit in [100, 200, 300, 400]:

    print('\nResolution: %d' % nr_fit)
    r_fit = cp.linspace(rl, ru, nr_fit)
    G_fit = Sphere.compute_scattering_intensity([q], [r_fit], drho)

    # noise-free data, so use relative error bars and no regularization
    t0 = time.time()
    xi_opt, b_opt, w_opt_list = optimize(G_fit, I_data, I_data, sigma=None)
    t1 = time.time()
    print('Inversion time: %.2f s' % (t1-t0))

    # plot optimized distribution
    plot_sphere_distribution(r_fit.get(), w_opt_list[0],
                             title='Optimized distribution (%d radii)' % nr_fit)

    # compute model intensities and chi^2
    Gw_opt = contract_tensor(G_fit, [cp.asarray(w_opt_list[0])], skip_axes=[0])
    I_opt = xi_opt * Gw_opt + b_opt
    chi2 = cp.sum(((I_opt - I_data) / I_data) ** 2)
    print('Chi^2: %.5e' % chi2)

    # plot optimized intensities
    plot_1d_optimized_intensity(q.get(), I_data.get(), I_opt.get())
