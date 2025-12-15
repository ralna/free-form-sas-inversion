"""
Low-rank Sphere Green's function approximation
https://www.sasview.org/docs/user/models/sphere.html

Parameters:
q - scattering vector
r - sphere radius
drho - difference between scattering length densities

Copyright (C) 2025 The Science and Technology Facilities Council (STFC)
"""
import numpy as np
from ffsi.models.sphere import G_sphere
from ...greedy.cross import greedy_cross

# for timing
import time

# contrast
drho = 1

# q discretisation
ql = 1e-3
qu = 1
nq = 200

# r discretisation
rl = 400
ru = 800
nr = 200

# discretise q and r
q = np.linspace(ql, qu, nq)
r = np.linspace(rl, ru, nr)

# function for cross-approximation
# TODO: check if SASView has any vectorisation
dims = (nq,nr)
def G_func(inds):
    n = inds.shape[0]
    g_vals = np.zeros(n)
    for i in range(n):
        g_vals[i] = G_sphere(q[inds[i,0]], r[inds[i,1]], drho)
    return g_vals

# form low-rank TT-representation
tol = 1e-6
nswp = 1000
print('Computing TT-representation using greedy cross...')
print('Tolerance: %.2e' % tol)
print('Max sweeps: %d' % nswp)

t0 = time.time()
cores = greedy_cross(dims, G_func, tol, nswp)
t1 = time.time()
print('Greedy cross time %.2f s' % (t1-t0))

print('Number of cores: %d' % len(cores))
print('Core sizes:')
for i in range(len(cores)):
    print(cores[i].shape)
