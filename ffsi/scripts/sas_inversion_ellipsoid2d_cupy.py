"""
Free-form SAS Inversion Script for Simulated Ellipsoid Data

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import time
import cupy as cp
from ffsi.models.ellipsoid2d import Ellipsoid2D
from ffsi.optimize_galahad import optimize
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
nqx = 60
nqy = 60
qx = cp.linspace(-0.75, 0.75, nqx)
qy = cp.linspace(-0.75, 0.75, nqy)

# rp discretisation
rpl = 50
rpu = 90
nrp = 18

# re discretisation
rel = 200
reu = 600
nre = 17

# theta discretisation
thetal = 5
thetau = 60
ntheta = 16

# phi discretisation
phil = 150
phiu = 240
nphi = 15

# discretise rp, re, theta, phi
rp = cp.linspace(rpl, rpu, nrp)
re = cp.linspace(rel, reu, nre)
theta = cp.linspace(thetal, thetau, ntheta)
phi = cp.linspace(phil, phiu, nphi)

print('qx: %d' % nqx)
print('qy: %d' % nqy)
print('rp: linspace(%d,%d,%d)' % (rpl, rpu, nrp))
print('re: linspace(%d,%d,%d)' % (rel, reu, nre))
print('theta: linspace(%d,%d,%d)' % (thetal, thetau, ntheta))
print('phi: linspace(%d,%d,%d)' % (phil, phiu, nphi))

## Step 1: Generate ground truth
print("\nStep 1: Generate ground truth\n")

# generate ground truth distributions
w_rp_true = crazy_distribution(rp, [(1, 60, 3), (2, 70, 4), (2, 80, 3)], 1, 1, 1)
w_re_true = crazy_distribution(re, [(1.5, 300, 20), (1, 400, 20), (2, 500, 20)], 1, 1, 1)
w_theta_true = crazy_distribution(theta, [(4, 15, 5), (2, 35, 5), (2, 50, 5)], 2, 1, 1)
w_phi_true = crazy_distribution(phi, [(2, 170, 10), (2, 200, 10), (4, 220, 10)], 3, 1, 1)

# convert degrees to radians
theta = cp.deg2rad(theta)
phi = cp.deg2rad(phi)

# plot "true" distributions
plot_ellipsoid2d_distributions([rp.get(),re.get(),theta.get(),phi.get()], [w_rp_true.get(),w_re_true.get(),w_theta_true.get(),w_phi_true.get()], title='True distributions')

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

# Calculate required memory for G
print('\nG tensor memory requirements:')
G_elem = nqx * nqy * nrp * nre * ntheta * nphi
bits = cp.finfo(cp.dtype(float)).bits
G_mem = (G_elem * bits) / 8e9
print(f'G elements: {G_elem:,}')
print('G memory: %.2f GB' % G_mem)

# Compute true G
print('\nComputing full G tensor on GPU...')
q_list = [qx,qy]
param_list = [rp, re, theta, phi]
t0 = time.time()
G = sasmodel.compute_scattering_intensity(q_list, param_list, drho)
t1 = time.time()
print('G computation time on GPU: %.2f s' % (t1-t0))

# compute Gw_true for simulating the intensities
print('\nComputing Gw for simulating the intensities...', end='')
w_true_list = [w_rp_true, w_re_true, w_theta_true, w_phi_true]
Gw_true = contract_tensor(G, w_true_list, skip_axes=[0,1])
print('done.')

# compute model intensities as the intensity data
I_data = xi_true * Gw_true + b_true

# plot intensities
plot_2d_intensity(qx.get(), qy.get(), I_data.get())

## Step 2: SAS Inversion with true G
print("\nStep 2: SAS inversion with true G\n")
xi_opt, b_opt, w_opt_list = optimize(G, I_data, I_data, sigma=None)

# plot optimized distributions
plot_ellipsoid2d_distributions([rp.get(),re.get(),theta.get(),phi.get()], w_opt_list, title='Optimized distributions')

# Transfer optimized distributions to GPU
w_opt_list_gpu = [cp.asarray(w) for w in w_opt_list]

# compute Gw_opt for the optimized intensities
print('\nComputing Gw for the optimized intensities...', end='')
Gw_opt = contract_tensor(G, w_opt_list_gpu, skip_axes=[0,1])
print('done.')

# compute model intensities as the intensity data
I_opt = xi_opt * Gw_opt + b_opt

# plot optimized intensities
plot_2d_optimized_intensity(qx.get(), qy.get(), I_data.get(), I_opt.get())

# plot intensity misfit
plot_2d_intensity_misfit(qx.get(), qy.get(), I_data.get(), I_opt.get())
