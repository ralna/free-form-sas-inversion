"""
Free-form SAS Inversion Script for Simulated Ellipsoid Data (using full G tensor)

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import numpy as np
from ffsi.models.numpy.ellipsoid import G_ellipsoid
from ffsi.optimize_ellipsoid_galahad_tensor import tt_optimize

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
qx = np.linspace(-0.75, 0.75, nqx)
qy = np.linspace(-0.75, 0.75, nqy)

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

# ground truth distributions generator from paper
def crazy_distribution(x, gaussians, noise_level, fade_start, fade_end, seed=0):
    # create
    w_true = np.zeros(x.shape)

    # add Gaussians
    for factor, mean, stddev in gaussians:
        w_true += factor * np.exp(-((x - mean) / stddev) ** 2)

    # add noise
    np.random.seed(seed)
    w_true += noise_level * np.random.rand(*x.shape) * np.random.rand(*x.shape)

    # fade both ends to make it look nicer
    if len(x) >= 3:
        w_true[0:fade_start] = 0.
        w_true[fade_start:fade_end] *= np.linspace(0, 1, fade_end - fade_start)
        w_true[-fade_start:] = 0.
        w_true[-fade_end:-fade_start] *= np.linspace(1, 0, fade_end - fade_start)

    # normalize to 1
    w_true /= np.sum(w_true)
    return w_true

# generate ground truth distributions
w_rp_true = crazy_distribution(rp, [(1, 60, 3), (2, 70, 4), (2, 80, 3)], 1, 1, 1)
w_re_true = crazy_distribution(re, [(1.5, 300, 20), (1, 400, 20), (2, 500, 20)], 1, 1, 1)
w_theta_true = crazy_distribution(theta, [(4, 15, 5), (2, 35, 5), (2, 50, 5)], 2, 1, 1)
w_phi_true = crazy_distribution(phi, [(2, 170, 10), (2, 200, 10), (4, 220, 10)], 3, 1, 1)

# convert degrees to radians
theta = np.deg2rad(theta)
phi = np.deg2rad(phi)

# plot "true" distributions
fig, ax = plt.subplots(2, 2)
plt.suptitle("True distributions")
plt.subplots_adjust(hspace=.5, wspace=.5)
ax[0,0].plot(rp, w_rp_true * 100) # x100 to percent
ax[0,1].plot(re, w_re_true * 100) # x100 to percent
ax[1,0].plot(theta, w_theta_true * 100) # x100 to percent
ax[1,1].plot(phi, w_phi_true * 100) # x100 to percent
ax[0,0].set_xlabel(r"Polar radius $r_p$ ($\AA$)")
ax[0,1].set_xlabel(r"Equatorial radius $r_e$ ($\AA$)")
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
V = 4/3 * np.pi * rp[:,np.newaxis] * re[np.newaxis,:] ** 2 # ellipsoid volume
V_ave = w_rp_true.T @ V @ w_re_true
xi_true = 1e-4 * scale_true / V_ave
print('xi_true: %.2e' % xi_true)

# Compute true G
print('\nComputing full G tensor on CPU...')
dims = (nqx,nqy,nrp,nre,ntheta,nphi)
G = G_ellipsoid(qx, qy, rp, re, theta, phi, drho)

# compute Gw_true for simulating the intensities
print('\nComputing Gw for simulating the intensities...')
Gw_true = np.tensordot(np.tensordot(np.tensordot(np.tensordot(G, w_rp_true, axes=(2,0)), w_re_true, axes=(2,0)), w_theta_true, axes=(2,0)), w_phi_true, axes=(2,0))

# compute model intensities as the intensity data
I_data = xi_true * Gw_true + b_true

# plot intensities
plt.figure()
plt.imshow(I_data.T,
           extent=(qx[0], qx[-1], qy[0], qy[-1]), aspect=1., cmap='turbo',
           norm=colors.LogNorm(vmin=I_data.min(), vmax=I_data.max()))
plt.xlabel(r"Scattering vector $qx$ ($\AA^{-1}$)")
plt.ylabel(r"Scattering vector $qy$ ($\AA^{-1}$)")
plt.title(r"Intensity image $I(q_x, q_y)$ ($\mathrm{cm}^{-1})$")
plt.colorbar()
plt.show()

## Step 2: SAS Inversion with true G
print("\nStep 2: SAS inversion with true G\n")
xi_opt, b_opt, w_rp_opt, w_re_opt, w_theta_opt, w_phi_opt = tt_optimize(G, dims, I_data, I_data, sigma=None,
                                                                      check_residual=False, check_derivative=False, xi_true=xi_true, b_true=b_true,
                                                                      w_rp_true=w_rp_true, w_re_true=w_re_true, w_theta_true=w_theta_true, w_phi_true=w_phi_true)

# plot optimized distributions
fig, ax = plt.subplots(2, 2)
plt.suptitle("Optimized distributions")
plt.subplots_adjust(hspace=.5, wspace=.5)
ax[0,0].plot(rp, w_rp_opt * 100) # x100 to percent
ax[0,1].plot(re, w_re_opt * 100) # x100 to percent
ax[1,0].plot(theta, w_theta_opt * 100) # x100 to percent
ax[1,1].plot(phi, w_phi_opt * 100) # x100 to percent
ax[0,0].set_xlabel(r"Polar radius $r_p$ ($\AA$)")
ax[0,1].set_xlabel(r"Equatorial radius $r_e$ ($\AA$)")
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
