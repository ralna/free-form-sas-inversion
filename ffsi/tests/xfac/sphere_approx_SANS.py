"""
Low-rank Sphere Green's function approximation (SANS Data)
https://www.sasview.org/docs/user/models/sphere.html

Parameters:
q - scattering vector
r - sphere radius
drho - difference between scattering length densities

Copyright (C) 2025 The Science and Technology Facilities Council (STFC)
"""
import numpy as np
from ffsi.models.sphere import G_sphere

# import locally compiled xfac python module
import sys
sys.path.append("../xfac/build/python")
import xfacpy

# for timing
import time

# load real SAXS data
data = np.loadtxt('ffsi/data/SANS/observation.txt')
q = data[:,0]
nq = len(q)

# contrast
drho = 1

# r discretisation
rl = 400
ru = 800
nr = 1000
r = np.linspace(rl, ru, nr)

# function for cross-interpolation
dims = (nq,nr)
G_func = lambda inds: G_sphere(q[inds[0]], r[inds[1]], drho)

# form low-rank TT-representation
tol = 1e-8
max_rank = 250
print('Computing TT-representation using xfac...')
print('Tolerance: %.2e' % tol)

t0 = time.time()
param = xfacpy.TensorCI2Param()
param.reltol = tol
param.bondDim = max_rank
tci = xfacpy.TensorCI2(G_func, dims, param=param)
while not tci.isDone():
    tci.iterate()
t1 = time.time()
print('xfac time: %.2f s' % (t1-t0))

rel_err = tci.pivotError[-1] / tci.pivotError[0]
print('xfac relative error: %.2e' % rel_err)
ncores = tci.len()
print('Number of cores: %d' % ncores)
print('Core sizes:')
for i in range(ncores):
    print(tci.tt.core[i].shape)

# FIXME: the below is for error estimation only
print('Forming G for TT-approximation error...')

# form Green's function tensor
G = np.zeros((nq,nr))
for iq in range(nq):
    for ir in range(nr):
        G[iq,ir] = G_sphere(q[iq], r[ir], drho)

# compute TT approximation error
abs_err = tci.trueError(max_n_eval=int(1e15))
rel_err = abs_err / np.linalg.norm(G)
print('TT-approximation relative error: %.2e' % rel_err)
