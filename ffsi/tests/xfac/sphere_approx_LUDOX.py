"""
Low-rank Sphere Green's function approximation (LUDOX Data)
https://www.sasview.org/docs/user/models/sphere.html

Parameters:
q - scattering vector
r - sphere radius
drho - difference between scattering length densities

Copyright (C) 2025 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import numpy as np
from ffsi.models.sphere import G_sphere
from ffsi.tensor_train import tt_approx

# load real LUDOX data
data = np.loadtxt('ffsi/data/LUDOX/S49_Ludox6_1pct.dat', skiprows=1)
q = data[:,0]
nq = len(q)

# contrast
drho = 1

# r discretisation (log)
rl = 0
ru = 2.5
nr = 1000
r = np.logspace(rl, ru, nr)

# function for cross-interpolation
dims = (nq,nr)
G_func = lambda inds: G_sphere(q[inds[0]], r[inds[1]], drho)

# form low-rank TT-representation
tt_approx(G_func, dims, tol=1e-6, max_rank=250, compute_true_error=True, max_error_evals=1e15)
