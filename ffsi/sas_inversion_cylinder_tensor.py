"""
Free-form SAS Inversion Script for Simulated Cylinder Data (using full G tensor)

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import numpy as np
from ffsi.models.cylinder import G_cylinder
from ffsi.optimize_cylinder_galahad_tensor import tt_optimize

# for plotting
import matplotlib.pyplot as plt
import matplotlib.colors as colors

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

# l discretisation
ll = 200
lu = 600
nl = 10

# r discretisation
rl = 50
ru = 90
nr = 9

# theta discretisation
thetal = 20
thetau = 75
ntheta = 8

# phi discretisation
phil = 150
phiu = 240
nphi = 7

# discretise l, r, theta, phi
l = np.linspace(ll, lu, nl)
r = np.linspace(rl, ru, nr)
theta = np.linspace(thetal, thetau, ntheta)
phi = np.linspace(phil, phiu, nphi)

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
w_l_true = crazy_distribution(l, [(1.5, 300, 20), (1, 400, 20), (2, 500, 20)], 0, 1, 1)
w_r_true = crazy_distribution(r, [(1, 60, 3), (2, 70, 4), (2, 80, 3)], 0, 1, 1)
w_theta_true = crazy_distribution(theta, [(4, 30, 5), (2, 50, 5), (2, 65, 5)], 0, 1, 1)
w_phi_true = crazy_distribution(phi, [(2, 170, 10), (2, 200, 10), (4, 220, 10)], 0, 1, 1)

# convert degrees to radians
theta = np.deg2rad(theta)
phi = np.deg2rad(phi)

# plot "true" distributions
fig, ax = plt.subplots(2, 2)
plt.suptitle("True distributions")
plt.subplots_adjust(hspace=.5, wspace=.5)
ax[0,0].plot(l, w_l_true * 100) # x100 to percent
ax[0,1].plot(r, w_r_true * 100) # x100 to percent
ax[1,0].plot(theta, w_theta_true * 100) # x100 to percent
ax[1,1].plot(phi, w_phi_true * 100) # x100 to percent
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
V = np.pi * l[:,np.newaxis] * r[np.newaxis,:] ** 2 # cylinder volume
V_ave = w_l_true.T @ V @ w_r_true
xi_true = 1e-4 * scale_true / V_ave
print('xi_true: %.2e' % xi_true)

# Compute true G
print('\nComputing full G tensor...')
dims = (nqx,nqy,nl,nr,ntheta,nphi)
G = np.zeros(dims)
for iqx in range(nqx):
    print('  progress at iqx %d out of %d' % (iqx+1,nqx))
    for iqy in range(nqy):
        for il in range(nl):
            for ir in range(nr):
                for it in range(ntheta):
                    for ip in range(nphi):
                        G[iqx,iqy,il,ir,it,ip] = G_cylinder(qx[iqx], qy[iqy], l[il], r[ir], theta[it], phi[ip], drho)

# compute Gw_true for simulating the intensities
print('\nComputing Gw for simulating the intensities...')
Gw_true = np.zeros((nqx,nqy))
for iqx in range(nqx):
    print('  progress at iqx %d out of %d' % (iqx+1,nqx))
    for iqy in range(nqy):
        for il in range(nl):
            for ir in range(nr):
                for it in range(ntheta):
                    for ip in range(nphi):
                        Gw_true[iqx,iqy] += G[iqx,iqy,il,ir,it,ip] * w_l_true[il] * w_r_true[ir] * w_theta_true[it] * w_phi_true[ip]

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
xi_opt, b_opt, w_l_opt, w_r_opt, w_theta_opt, w_phi_opt = tt_optimize(G, dims, I_data, I_data, sigma=None,
                                                                      check_residual=False, check_derivative=False, xi_true=xi_true, b_true=b_true,
                                                                      w_l_true=w_l_true, w_r_true=w_r_true, w_theta_true=w_theta_true, w_phi_true=w_phi_true)

# plot optimized distributions
fig, ax = plt.subplots(2, 2)
plt.suptitle("Optimized distributions")
plt.subplots_adjust(hspace=.5, wspace=.5)
ax[0,0].plot(l, w_l_opt * 100) # x100 to percent
ax[0,1].plot(r, w_r_opt * 100) # x100 to percent
ax[1,0].plot(theta, w_theta_opt * 100) # x100 to percent
ax[1,1].plot(phi, w_phi_opt * 100) # x100 to percent
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
