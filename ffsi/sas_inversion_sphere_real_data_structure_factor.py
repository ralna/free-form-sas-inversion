"""
Free-form SAS Inversion Script for Real Sphere Data with Structure Factor (LUDOX)

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
"""
import numpy as np
from ffsi.models.sphere import G_sphere
from ffsi.models.hardsphere import S_hard_sphere
from ffsi.tensor_train import tt_approx
from ffsi.optimize_structure_factor import tt_optimize_structure_factor

# for plotting
import matplotlib.pyplot as plt

## Step 0: Choose dataset
dataset = 'S49_Ludox6_1pct.dat' # options are in data/LUDOX

## Step 1: Load real data and discretise parameters
print("Step 1: Load data and discretise parameters\n")

# load LUDOX data
data = np.loadtxt('ffsi/data/LUDOX/'+dataset, skiprows=1)

# extract parameters
tr = 60  # truncate at low-q
q = data[tr:,0]
I_data = data[tr:,1]
I_data_std = data[tr:,2]
nq = len(q)

# r discretisation (log)
rl = 0
ru = 2.5
nr = 1000
r = np.logspace(rl, ru, nr)

# contrast
drho = 1

print('q: fromdata(%d,%d,%d)' % (min(q), max(q), nq))
print('r: logspace(%d,%d,%d)' % (rl, ru, nr))

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

## Step 2: Form structure factor
print("\nStep 2: Forming Structure Factor\n")

# function for structure factor
S_func = lambda iq, r_eff, vol_frac: S_hard_sphere(q[iq], r_eff, vol_frac)

# FIXME: this is just for plotting the structure factor
print('\nForming S for plotting...')
S = np.zeros(nq)
for iq in range(nq):
    S[iq] = S_hard_sphere(q[iq], r_eff=240, vol_frac=0.07)

# plot structure factor
plt.figure()
plt.plot(q, S)
plt.grid()
plt.xscale('log')
plt.xlabel(r"Scattering vector $q$ ($\AA^{-1}$)")
plt.ylabel(r"Structure factor $S$")
plt.show()

## Step 3: Approximate Green's function
print("\nStep 3: Approximate Green's function\n")

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

## Step 4: SAS Inversion with low-rank G
print("\nStep 4: SAS inversion with low-rank G\n")
xi_opt, b_opt, r_eff_opt, vol_frac_opt, w_opt = tt_optimize_structure_factor(tt, dims, I_data, I_data_std, S_func, r_eff_0=180, vol_frac_0=0.05)

# plot optimized distributions
plt.figure()
plt.plot(r, w_opt)
plt.grid()
plt.title("Optimized distributions")
plt.xlabel(r"Radius $r$ ($\AA$)")
plt.ylabel(r"Weights $w$")
plt.show()
