"""
Free-form SAS Inversion Script for Simulated Ellipsoid Data

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import numpy as np
from ffsi.models.ellipsoid import G_ellipsoid
from ffsi.tensor_train import tt_approx
from ffsi.optimize_ellipsoid_galahad import tt_optimize

# for plotting
import matplotlib.pyplot as plt
import matplotlib.colors as colors

## Step 0: Discretisation parameters
print("Step 0: Discretise parameters\n")

# contrast
drho = 1

# qx, qy discretisation
nqx = 120
nqy = 120
q_side = np.logspace(-2, 0, 50) # log scale on the sides
q_center = np.linspace(-0.0095, 0.0095, 20) # linear scale in the cente
qx = np.hstack((-q_side[::-1], q_center, q_side))
qy = qx.copy()

# rp discretisation
rpl = 50
rpu = 90
nrp = 100

# re discretisation
rel = 200
reu = 600
nre = 100

# theta discretisation
thetal = 20
thetau = 75
ntheta = 100

# phi discretisation
phil = 150
phiu = 240
nphi = 100

# discretise rp, re, theta, phi
rp = np.linspace(rpl, rpu, nrp)
re = np.linspace(rel, reu, nre)
theta = np.linspace(thetal, thetau, ntheta)
phi = np.linspace(phil, phiu, nphi)

# convert degrees to radians
theta = np.deg2rad(theta)
phi = np.deg2rad(phi)

print('qx: %d' % nqx)
print('qy: %d' % nqy)
print('rp: linspace(%d,%d,%d)' % (rpl, rpu, nrp))
print('re: linspace(%d,%d,%d)' % (rel, reu, nre))
print('theta: linspace(%d,%d,%d)' % (thetal, thetau, ntheta))
print('phi: linspace(%d,%d,%d)' % (phil, phiu, nphi))

## Step 1: Load pre-generated ground truth
print("\nStep 1: Load pre-generated ground truth\n")

# load true distributions
w_rp_true = np.loadtxt('ffsi/data/ellipsoid_very_large/w_rp_true.txt')
w_re_true = np.loadtxt('ffsi/data/ellipsoid_very_large/w_re_true.txt')
w_theta_true = np.loadtxt('ffsi/data/ellipsoid_very_large/w_theta_true.txt')
w_phi_true = np.loadtxt('ffsi/data/ellipsoid_very_large/w_phi_true.txt')

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
ax[1,0].set_xlabel(r"Ellipsoid axis to beam angle $\theta$ (radians)")
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
V = 4/3 * np.pi * rp[:,np.newaxis] * re[np.newaxis,:] ** 2 # ellipsoid volume
V_ave = w_rp_true.T @ V @ w_re_true
xi_true = 1e-4 * scale_true / V_ave
print('xi_true: %.2e' % xi_true)

# load true intensity data
I_data = np.loadtxt('ffsi/data/ellipsoid_very_large/intensities.txt')

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

## Step 2: Approximate Green's function
print("\nStep 2: Approximate Green's function\n")

# function for cross-interpolation
dims = (nqx,nqy,nrp,nre,ntheta,nphi)
G_func = lambda inds: G_ellipsoid(qx[inds[0]], qy[inds[1]], rp[inds[2]], re[inds[3]], theta[inds[4]], phi[inds[5]], drho)

# form low-rank TT-representation
tt = tt_approx(G_func, dims, tol=1e-7, max_rank=250, compute_true_error=False)

## Step 3: SAS Inversion with low-rank G
print("\nStep 3: SAS inversion with low-rank G\n")
xi_opt, b_opt, w_rp_opt, w_re_opt, w_theta_opt, w_phi_opt = tt_optimize(tt, dims, I_data, I_data, sigma=0,
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
ax[1,0].set_xlabel(r"Ellipsoid axis to beam angle $\theta$ (radians)")
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
