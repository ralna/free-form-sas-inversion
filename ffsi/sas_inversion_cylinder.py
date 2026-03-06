"""
Free-form SAS Inversion Script for Simulated Cylinder Data

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
"""
import numpy as np
from ffsi.models.cylinder import G_cylinder
from ffsi.tensor_train import tt_approx
from ffsi.optimize_cylinder import tt_optimize

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

# l discretisation
ll = 200
lu = 600
nl = 40

# r discretisation
rl = 50
ru = 90
nr = 40

# theta discretisation
thetal = 20/180 * np.pi
thetau = 75/180 * np.pi
ntheta = 40

# phi discretisation
phil = 150/180 * np.pi
phiu = 240/180 * np.pi
nphi = 40

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

## Step 1: Load pre-generated ground truth
print("\nStep 1: Load pre-generated ground truth\n")

# load model intensity data from paper
I_data = np.loadtxt('ffsi/data/cylinder_large_scale/intensities.txt')

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
xi_opt, b_opt, w_l_opt, w_r_opt, w_theta_opt, w_phi_opt = tt_optimize(tt, dims, I_data, I_data)

# plot optimized distributions
fig, ax = plt.subplots(2, 2)
plt.suptitle("Optimized distributions")
plt.subplots_adjust(hspace=.5, wspace=.5)
ax[0,0].plot(l, w_l_opt)
ax[0,1].plot(r, w_r_opt)
ax[1,0].plot(theta, w_theta_opt)
ax[1,1].plot(phi, w_phi_opt)
ax[0,0].set_xlabel(r"Length $l$ ($\AA$)")
ax[0,1].set_xlabel(r"Radius $r$ ($\AA$)")
ax[1,0].set_xlabel(r"Cylinder axis to beam angle $\theta$ (radians)")
ax[1,1].set_xlabel(r"Rotation about beam $\phi$ (radians)")
ax[0,0].set_ylabel(r"Weights $w$")
ax[0,1].set_ylabel(r"Weights $w$")
ax[1,0].set_ylabel(r"Weights $w$")
ax[1,1].set_ylabel(r"Weights $w$")
ax[0,0].grid()
ax[0,1].grid()
ax[1,0].grid()
ax[1,1].grid()
plt.show()
