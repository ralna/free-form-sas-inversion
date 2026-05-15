"""
Free-form SAS Inversion Script for Simulated Cylinder Data

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import time
import cupy as cp
from ffsi.models.cylinder import G_cylinder
from ffsi.optimize_cylinder_galahad import tt_optimize

# for plotting
import matplotlib.pyplot as plt
import matplotlib.colors as colors

## Step 0: Discretisation parameters
print("Step 0: Discretise parameters\n")

# contrast
drho = 1

# qx, qy discretisation
nqx = 60
nqy = 60
qx = cp.linspace(-0.75, 0.75, nqx)
qy = cp.linspace(-0.75, 0.75, nqy)

# l discretisation
ll = 200
lu = 600
nl = 18

# r discretisation
rl = 50
ru = 90
nr = 17

# theta discretisation
thetal = 5
thetau = 60
ntheta = 16

# phi discretisation
phil = 150
phiu = 240
nphi = 15

# discretise l, r, theta, phi
l = cp.linspace(ll, lu, nl)
r = cp.linspace(rl, ru, nr)
theta = cp.linspace(thetal, thetau, ntheta)
phi = cp.linspace(phil, phiu, nphi)

print('qx: %d' % nqx)
print('qy: %d' % nqy)
print('l: linspace(%d,%d,%d)' % (ll, lu, nl))
print('r: linspace(%d,%d,%d)' % (rl, ru, nr))
print('theta: linspace(%d,%d,%d)' % (thetal, thetau, ntheta))
print('phi: linspace(%d,%d,%d)' % (phil, phiu, nphi))

## Step 1: Generate ground truth
print("\nStep 1: Generate ground truth\n")

# ground truth distributions generator from paper
def crazy_distribution(x, gaussians, noise_level, fade_start, fade_end, seed=0):
    # create
    w_true = cp.zeros(x.shape)

    # add Gaussians
    for factor, mean, stddev in gaussians:
        w_true += factor * cp.exp(-((x - mean) / stddev) ** 2)

    # add noise
    cp.random.seed(seed)
    w_true += noise_level * cp.random.rand(*x.shape) * cp.random.rand(*x.shape)

    # fade both ends to make it look nicer
    if len(x) >= 3:
        w_true[0:fade_start] = 0.
        w_true[fade_start:fade_end] *= cp.linspace(0, 1, fade_end - fade_start)
        w_true[-fade_start:] = 0.
        w_true[-fade_end:-fade_start] *= cp.linspace(1, 0, fade_end - fade_start)

    # normalize to 1
    w_true /= cp.sum(w_true)
    return w_true

# generate ground truth distributions
w_l_true = crazy_distribution(l, [(1.5, 300, 20), (1, 400, 20), (2, 500, 20)], 1, 1, 1)
w_r_true = crazy_distribution(r, [(1, 60, 3), (2, 70, 4), (2, 80, 3)], 1, 1, 1)
w_theta_true = crazy_distribution(theta, [(4, 15, 5), (2, 35, 5), (2, 50, 5)], 2, 1, 1)
w_phi_true = crazy_distribution(phi, [(2, 170, 10), (2, 200, 10), (4, 220, 10)], 3, 1, 1)

# convert degrees to radians
theta = cp.deg2rad(theta)
phi = cp.deg2rad(phi)

# plot "true" distributions
fig, ax = plt.subplots(2, 2)
plt.suptitle("True distributions")
plt.subplots_adjust(hspace=.5, wspace=.5)
ax[0,0].plot(l.get(), w_l_true.get() * 100) # x100 to percent
ax[0,1].plot(r.get(), w_r_true.get() * 100) # x100 to percent
ax[1,0].plot(theta.get(), w_theta_true.get() * 100) # x100 to percent
ax[1,1].plot(phi.get(), w_phi_true.get() * 100) # x100 to percent
ax[0,0].set_xlabel(r"Length $l$ ($\AA$)")
ax[0,1].set_xlabel(r"Radius $r$ ($\AA$)")
ax[1,0].set_xlabel(r"Cylinder axis to beam angle $\theta$ (radians)")
ax[1,1].set_xlabel(r"Rotation about beam $\phi$ (radians)")
ax[0,0].set_ylabel(r"Weights $w$ (%)")
ax[0,1].set_ylabel(r"Weights $w$ (%)")
ax[1,0].set_ylabel(r"Weights $w$ (%)")
ax[1,1].set_ylabel(r"Weights $w$ (%)")
ax[0,0].grid()
ax[0,1].grid()
ax[1,0].grid()
ax[1,1].grid()
plt.show()

# ground truth of scale and background
scale_true = 0.15
b_true = 2.2e-4
print('b_true: %.2e' % b_true)

# compute the ground truth of xi
# TODO: use a models function for the volume
V = cp.pi * l[:,cp.newaxis] * r[cp.newaxis,:] ** 2 # cylinder volume
V_ave = w_l_true.T @ V @ w_r_true
xi_true = 1e-4 * scale_true / V_ave
print('xi_true: %.2e' % xi_true)

# Calculate required memory for G
print('\nG tensor memory requirements:')
G_elem = nqx * nqy * nl * nr * ntheta * nphi
bits = cp.finfo(cp.dtype(float)).bits
G_mem = (G_elem * bits) / 8e9
print(f'G elements: {G_elem:,}')
print('G memory: %.2f GB' % G_mem)

# Compute true G
print('\nComputing full G tensor on GPU...')
dims = (nqx,nqy,nl,nr,ntheta,nphi)
t0 = time.time()
G = G_cylinder(qx, qy, l, r, theta, phi, drho)
t1 = time.time()
print('G computation time on GPU: %.2f s' % (t1-t0))

# compute Gw_true for simulating the intensities
print('\nComputing Gw for simulating the intensities...', end='')
Gw_true = (((G @ w_phi_true) @ w_theta_true) @ w_r_true) @ w_l_true
print('done.')

# compute model intensities as the intensity data
I_data = xi_true * Gw_true + b_true

# plot intensities
plt.figure()
plt.imshow(I_data.T.get(),
           extent=(qx[0].get(), qx[-1].get(), qy[0].get(), qy[-1].get()), aspect=1., cmap='turbo',
           norm=colors.LogNorm(vmin=I_data.min(), vmax=I_data.max()))
plt.xlabel(r"Scattering vector $qx$ ($\AA^{-1}$)")
plt.ylabel(r"Scattering vector $qy$ ($\AA^{-1}$)")
plt.title(r"Intensity image $I(q_x, q_y)$ ($\mathrm{cm}^{-1})$")
plt.colorbar()
plt.show()

## Step 2: SAS Inversion with true G
print("\nStep 2: SAS inversion with true G\n")
xi_opt, b_opt, w_l_opt, w_r_opt, w_theta_opt, w_phi_opt = tt_optimize(G, dims, I_data, I_data, sigma=None)

# plot optimized distributions
fig, ax = plt.subplots(2, 2)
plt.suptitle("Optimized distributions")
plt.subplots_adjust(hspace=.5, wspace=.5)
ax[0,0].plot(l.get(), w_l_opt * 100) # x100 to percent
ax[0,1].plot(r.get(), w_r_opt * 100) # x100 to percent
ax[1,0].plot(theta.get(), w_theta_opt * 100) # x100 to percent
ax[1,1].plot(phi.get(), w_phi_opt * 100) # x100 to percent
ax[0,0].set_xlabel(r"Length $l$ ($\AA$)")
ax[0,1].set_xlabel(r"Radius $r$ ($\AA$)")
ax[1,0].set_xlabel(r"Cylinder axis to beam angle $\theta$ (radians)")
ax[1,1].set_xlabel(r"Rotation about beam $\phi$ (radians)")
ax[0,0].set_ylabel(r"Weights $w$ (%)")
ax[0,1].set_ylabel(r"Weights $w$ (%)")
ax[1,0].set_ylabel(r"Weights $w$ (%)")
ax[1,1].set_ylabel(r"Weights $w$ (%)")
ax[0,0].grid()
ax[0,1].grid()
ax[1,0].grid()
ax[1,1].grid()
plt.show()

# Transfer optimized distributions to GPU
w_l_opt = cp.asarray(w_l_opt)
w_r_opt = cp.asarray(w_r_opt)
w_theta_opt = cp.asarray(w_theta_opt)
w_phi_opt = cp.asarray(w_phi_opt)

# compute Gw_opt for the optimized intensities
print('\nComputing Gw for the optimized intensities...', end='')
Gw_opt = (((G @ w_phi_opt) @ w_theta_opt) @ w_r_opt) @ w_l_opt
print('done.')

# compute model intensities as the intensity data
I_opt = xi_opt * Gw_opt + b_opt

# plot optimized intensities
plt.figure()
plt.imshow(I_opt.T.get(),
           extent=(qx[0].get(), qx[-1].get(), qy[0].get(), qy[-1].get()), aspect=1., cmap='turbo',
           norm=colors.LogNorm(vmin=I_data.min(), vmax=I_data.max()))
plt.xlabel(r"Scattering vector $qx$ ($\AA^{-1}$)")
plt.ylabel(r"Scattering vector $qy$ ($\AA^{-1}$)")
plt.title(r"Optimized Intensity image $I(q_x, q_y)$ ($\mathrm{cm}^{-1})$")
plt.colorbar()
plt.show()

# plot intensity misfit
plt.figure()
plt.imshow(cp.abs(I_data.T - I_opt.T).get(),
           extent=(qx[0].get(), qx[-1].get(), qy[0].get(), qy[-1].get()), aspect=1., cmap='turbo',
           norm=colors.LogNorm(vmin=I_data.min(), vmax=I_data.max()))
plt.xlabel(r"Scattering vector $qx$ ($\AA^{-1}$)")
plt.ylabel(r"Scattering vector $qy$ ($\AA^{-1}$)")
plt.title(r"Intensity Misfit ($\mathrm{cm}^{-1})$")
plt.colorbar()
plt.show()
