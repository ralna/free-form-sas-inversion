"""
Free-form SAS Inversion Script for Real Sphere Data (SANS/SAXS)

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import time
import cupy as cp
from ffsi.models import Sphere
from ffsi.optimize_galahad import optimize
from ffsi.utils import contract_tensor

# for plotting
from ffsi.plotting import *

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
plot_1d_intensity(q.get(), I_data.get(), I_data_std=I_data_std.get())

# Calculate required memory for G
print('\nG tensor memory requirements:')
G_elem = nq * nr
bits = cp.finfo(cp.dtype(float)).bits
G_mem = (G_elem * bits) / 8e9
print(f'G elements: {G_elem:,}')
print('G memory: %.2f GB' % G_mem)

# Compute true G
print('\nComputing full G tensor on GPU...')
q_list = [q]
param_list = [r]
t0 = time.time()
G = Sphere.compute_scattering_intensity(q_list, param_list, drho)
t1 = time.time()
print('G computation time on GPU: %.2f s' % (t1-t0))

## Step 2: SAS Inversion with true G
print("\nStep 2: SAS inversion with true G\n")
xi_opt, b_opt, w_opt_list = optimize(G, I_data, I_data_std, sigma=0.25)

# plot optimized distributions
plot_sphere_distribution(r.get(), w_opt_list[0], title='Optimized distribution', normalize_by_volume=True)

# Transfer optimized distributions to GPU
w_r_opt = cp.asarray(w_opt_list[0])

# compute Gw_opt for the optimized intensities
print('\nComputing Gw for the optimized intensities...', end='')
Gw_opt = contract_tensor(G, [w_r_opt], skip_axes=[0])
print('done.')

# compute model intensities as the intensity data
I_opt = xi_opt * Gw_opt + b_opt

# plot optimized intensities
plot_1d_optimized_intensity(q.get(), I_data.get(), I_opt.get(), I_data_std=I_data_std.get())
