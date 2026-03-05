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
from ffsi.tensor_train import tt_approx

# load real SANS data
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
tt_approx(G_func, dims, tol=1e-6, max_rank=250, compute_true_error=True, max_error_evals=1e15)
