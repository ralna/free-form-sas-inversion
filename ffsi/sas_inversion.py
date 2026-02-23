"""
Free-form SAS Inversion Script

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
"""
import numpy as np
from ffsi.models.sphere import G_sphere
from ffsi.tensor_train import tt_approx
from ffsi.optimize import tt_optimize

# for plotting
import matplotlib.pyplot as plt

## Step 1: Approximate Green's function
print("Step 1: Approximate Green's function\n")

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

# function for cross-interpolation
dims = (nq,nr)
G_func = lambda inds: G_sphere(q[inds[0]], r[inds[1]], drho)

# form low-rank TT-representation
tt = tt_approx(G_func, dims, tol=1e-10, compute_true_error=True)

# FIXME: this is just for plotting the singular values
print('\nForming G for singular value computation...')

# form Green's function tensor
G = np.zeros((nq,nr))
for iq in range(nq):
    for ir in range(nr):
        G[iq,ir] = G_sphere(q[iq], r[ir], drho)

# plot singular values of G
s = np.linalg.svd(G, compute_uv=False)
plt.figure()
plt.semilogy(s/s[0])
plt.grid()
plt.title("Singular Value Decay")
plt.xlabel('Singular Value Index')
plt.ylabel('Normalised Singular Value')
plt.show()

## Step 2: Generate ground truth
print("\nStep 2: Generate ground truth\n")

# ground truth radii distributions
gaussian = np.exp(-(r - 500)**2 / (2*10**2))
boltzmann = 0.7 * np.exp(-np.abs(r - 700) / 20)
w_true = gaussian + boltzmann
w_true /= w_true.sum()  # normalize

# plot ground truth
plt.figure()
plt.plot(r, w_true)
plt.grid()
plt.title("Ground truth distributions")
plt.xlabel(r"Radius $r$ ($\AA$)")
plt.ylabel(r"Weights $w$")
plt.show()

# ground truth of scale and background
scale_true = 2
b_true = 0.5
print('b_true:', b_true)

# compute the ground truth of xi
V = 4/3 * np.pi * r ** 3 # sphere volume
V_ave = V @ w_true
xi_true = 1e-4 * scale_true / V_ave
print('xi_true:', xi_true)

# compute intensities
I_data = xi_true * G @ w_true + b_true

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

## Step 3: SAS Inversion with low-rank G
print("\nStep 3: SAS inversion with low-rank G\n")
xi_opt, b_opt, w_opt = tt_optimize(tt, dims, I_data, I_data_std, check_residual=True, check_derivative=True, xi_true=xi_true, b_true=b_true, w_true=w_true)

# plot optimized distributions
plt.figure()
plt.plot(r, w_opt)
plt.grid()
plt.title("Optimized distributions")
plt.xlabel(r"Radius $r$ ($\AA$)")
plt.ylabel(r"Weights $w$")
plt.show()
