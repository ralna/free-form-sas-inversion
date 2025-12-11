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

# import locally compiled xfac python module
import sys
sys.path.append("../xfac/build/python")
import xfacpy

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

# function for cross-interpolation
dims = (nq,nr)
G_func = lambda inds: G_sphere(q[inds[0]], r[inds[1]], drho)

# form low-rank TT-representation
tol = 1e-6
nswp = 100
t0 = time.time()
tci = xfacpy.TensorCI2(G_func, dims)
while not tci.isDone():
    tci.iterate()
t1 = time.time()
print('Greedy cross time (s): ')
print(t1-t0)

# # FIXME: the below is for error estimation only
# print('Computing TT-approximation error...')

# # form Green's function tensor
# G = np.zeros((nq,nr))
# for iq in range(nq):
#     for ir in range(nr):
#         G[iq,ir] = G_sphere(q[iq], r[ir], drho)

# # compute TT approximation error
# E = fGC - G
# error = np.linalg.norm(E)/np.linalg.norm(G)
# print('TT-approximation relative error: %.2e' % error)
