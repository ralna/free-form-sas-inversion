"""
Free-form SAS Inversion Script for Real Data (SANS/SAXS)

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
"""
import numpy as np
from ffsi.models.sphere import G_sphere
from ffsi.tensor_train import tt_approx
from ffsi.optimize import tt_optimize

# for plotting
import matplotlib.pyplot as plt

## Step 0: Choose dataset
dataset = 'SANS' # options are SANS or SAXS

## Step 1: Load real data and discretise parameters
print("Step 1: Load data and discretise parameters\n")
if dataset == 'SANS':

    # load SANS data
    data = np.loadtxt('ffsi/data/SANS/observation.txt')

    # extract parameters
    tr = 285  # truncate at high-q (noisy)
    q = data[:tr,0]
    I_data = data[:tr,1]
    I_data_std = data[:tr,2]
    nq = len(q)

    # r discretisation
    rl = 400
    ru = 800
    nr = 1000
    r = np.linspace(rl, ru, nr)

elif dataset == 'SAXS':

    # load SAXS data
    data = np.loadtxt('ffsi/data/SAXS/observation_corrected.txt')

    # extract parameters
    q = data[:,0]
    I_data = data[:,1]
    I_data_std = data[:,2]
    nq = len(q)

    # r discretisation
    rl = 400
    ru = 1200
    nr = 1000
    r = np.linspace(rl, ru, nr)

# contrast
drho = 1

print('q: fromdata(%d,%d,%d)' % (min(q), max(q), nq))
print('r: linspace(%d,%d,%d)' % (rl, ru, nr))

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

## Step 2: Approximate Green's function
print("\nStep 2: Approximate Green's function\n")

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

## Step 3: SAS Inversion with low-rank G
print("\nStep 3: SAS inversion with low-rank G\n")
xi_opt, b_opt, w_opt = tt_optimize(tt, dims, I_data, I_data_std)

# plot optimized distributions
plt.figure()
plt.plot(r, w_opt)
plt.grid()
plt.title("Optimized distributions")
plt.xlabel(r"Radius $r$ ($\AA$)")
plt.ylabel(r"Weights $w$")
plt.show()
