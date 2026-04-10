"""
Free-form SAS Inversion Script for Simulated Cylinder Data

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import numpy as np
from ffsi.models.cylinder import G_cylinder
from ffsi.tensor_train import tt_approx
#from ffsi.optimize_cylinder_galahad import tt_optimize
from ffsi.optimize_cylinder_nonlinear_squared import tt_optimize

# for plotting
import matplotlib.pyplot as plt
import matplotlib.colors as colors

## Step 0: Discretisation parameters
print("Step 0: Discretise parameters\n")

# contrast
drho = 1

# qx discretisation
qxl = -0.75
qxu = 0.75
nqx = 60

# qy discretisation
qyl = -0.75
qyu = 0.75
nqy = 60

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

# discretise q, l, r, theta, phi
qx = np.linspace(qxl, qxu, nqx)
qy = np.linspace(qyl, qyu, nqy)
l = np.linspace(ll, lu, nl)
r = np.linspace(rl, ru, nr)
theta = np.linspace(thetal, thetau, ntheta)
phi = np.linspace(phil, phiu, nphi)

# convert degrees to radians
theta = np.deg2rad(theta)
phi = np.deg2rad(phi)

print('qx: %d' % nqx)
print('qy: %d' % nqy)
print('l: linspace(%d,%d,%d)' % (ll, lu, nl))
print('r: linspace(%d,%d,%d)' % (rl, ru, nr))
print('theta: linspace(%d,%d,%d)' % (thetal, thetau, ntheta))
print('phi: linspace(%d,%d,%d)' % (phil, phiu, nphi))

## Step 1: Load pre-generated ground truth
print("\nStep 1: Load pre-generated ground truth\n")

# load true distributions from paper
w_l_true = np.loadtxt('ffsi/data/cylinder_small/w_l_true.txt')
w_r_true = np.loadtxt('ffsi/data/cylinder_small/w_r_true.txt')
w_theta_true = np.deg2rad(np.loadtxt('ffsi/data/cylinder_small/w_theta_true.txt'))
w_phi_true = np.deg2rad(np.loadtxt('ffsi/data/cylinder_small/w_phi_true.txt'))

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

# load true intensity data from paper
I_data = np.loadtxt('ffsi/data/cylinder_small/intensities.txt')

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
dims = (nqx,nqy,nl,nr,ntheta,nphi)
G_func = lambda inds: G_cylinder(qx[inds[0]], qy[inds[1]], l[inds[2]], r[inds[3]], theta[inds[4]], phi[inds[5]], drho)

# form low-rank TT-representation
tt = tt_approx(G_func, dims, tol=1e-10, max_rank=250, compute_true_error=False)

## Step 3: SAS Inversion with low-rank G
print("\nStep 3: SAS inversion with low-rank G\n")
xi_opt, b_opt, w_l_opt, w_r_opt, w_theta_opt, w_phi_opt = tt_optimize(tt, dims, I_data, I_data, check_residual=False, check_derivative=False, xi_true=xi_true, b_true=b_true,
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
