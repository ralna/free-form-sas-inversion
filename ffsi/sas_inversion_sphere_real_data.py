"""
Free-form SAS Inversion Script for Real Sphere Data (SANS/SAXS)

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import time
import cupy as cp
from ffsi.models.sphere import G_sphere
from ffsi.optimize_sphere_galahad import tt_optimize

# for plotting
import matplotlib.pyplot as plt

## Step 0: Choose dataset
dataset = 'SANS' # options are SANS or SAXS

## Step 1: Load real data and discretise parameters
print("Step 1: Load data and discretise parameters\n")
if dataset == 'SANS':

    # load SANS data
    data = cp.loadtxt('ffsi/data/SANS/observation.txt')

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
    r = cp.linspace(rl, ru, nr)

elif dataset == 'SAXS':

    # load SAXS data
    data = cp.loadtxt('ffsi/data/SAXS/observation_corrected.txt')

    # extract parameters
    q = data[:,0]
    I_data = data[:,1]
    I_data_std = data[:,2]
    nq = len(q)

    # r discretisation
    rl = 400
    ru = 1200
    nr = 1000
    r = cp.linspace(rl, ru, nr)

# contrast
drho = 1

print('q: fromdata(%d,%d,%d)' % (min(q), max(q), nq))
print('r: linspace(%d,%d,%d)' % (rl, ru, nr))

# plot intensities
plt.figure()
plt.errorbar(q.get(), I_data.get(), yerr=I_data_std.get(), ecolor='gray')
plt.grid()
plt.xscale('log')
plt.yscale('log')
plt.title('Intensity')
plt.xlabel(r"Scattering vector $q$ ($\AA^{-1}$)")
plt.ylabel(r"Intensity $I$ ($\mathrm{cm}^{-1}$)")
plt.show()

# Calculate required memory for G
print('\nG tensor memory requirements:')
G_elem = nq * nr
bits = cp.finfo(cp.dtype(float)).bits
G_mem = (G_elem * bits) / 8e9
print(f'G elements: {G_elem:,}')
print('G memory: %.2f GB' % G_mem)

# Compute true G
print('\nComputing full G tensor on GPU...')
dims = (nq,nr)
t0 = time.time()
G = G_sphere(q, r, drho)
t1 = time.time()
print('G computation time on GPU: %.2f s' % (t1-t0))

## Step 2: SAS Inversion with true G
print("\nStep 2: SAS inversion with true G\n")
xi_opt, b_opt, w_r_opt = tt_optimize(G, dims, I_data, I_data_std, sigma=0.25)

# Transfer optimized distributions to GPU
w_r_opt = cp.asarray(w_r_opt)

# plot optimized distributions
plt.figure()
v = r ** 3 # volume
w_r_hat = w_r_opt * v / (w_r_opt * v).sum() * 100  # x100 to percent
cmap = plt.get_cmap('turbo_r') # colormap
plt.plot(r.get(), w_r_hat.get(), c=cmap(0.0))
plt.grid()
plt.title("Optimized distributions")
plt.xlabel(r"Radius $r$ ($\AA$)")
plt.ylabel(r"Volume weight $\hat{w}$ (%)")
plt.show()

# compute Gw_opt for the optimized intensities
print('\nComputing Gw for the optimized intensities...', end='')
Gw_opt = G @ w_r_opt
print('done.')

# compute model intensities as the intensity data
I_opt = xi_opt * Gw_opt + b_opt

# plot optimized intensities
plt.figure()
plt.grid()
plt.errorbar(q.get(), I_data.get(), yerr=I_data_std.get(), ecolor='gray', marker='o', markerfacecolor='none')
plt.plot(q.get(), I_opt.get(), color='red', zorder=5)
plt.xscale('log')
plt.yscale('log')
plt.title('Optimized Intensity')
plt.xlabel(r"Scattering vector $q$ ($\AA^{-1}$)")
plt.ylabel(r"Intensity $I$ ($\mathrm{cm}^{-1}$)")
plt.legend(['Fit','Data'])
plt.show()
