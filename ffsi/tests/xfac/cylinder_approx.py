"""
Low-rank Cylinder Green's function approximation
https://www.sasview.org/docs/user/models/cylinder.html

Parameters:
qx - scattering vector x component
qy - scattering vector y component
l - cylinder length
r - cylinder radius
theta - cylinder axis to beam angle
phi - cylinder rotation about beam
drho - difference between scattering length densities

Copyright (C) 2025 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import numpy as np
from ffsi.models.cylinder import G_cylinder
from ffsi.tensor_train import tt_approx

# for timing
import time

# contrast
drho = 1

# qx discretisation
qxl = -0.75
qxu = 0.75
nqx = 10

# qy discretisation
qyl = -0.75
qyu = 0.75
nqy = 10

# l discretisation
ll = 200
lu = 600
nl = 10

# r discretisation
rl = 50
ru = 90
nr = 10

# theta discretisation
thetal = 5
thetau = 60
ntheta = 10

# phi discretisation
phil = 150
phiu = 240
nphi = 10

# discretise q, l, r, theta, phi
qx = np.linspace(qxl, qxu, nqx)
qy = np.linspace(qyl, qyu, nqy)
l = np.linspace(ll, lu, nl)
r = np.linspace(rl, ru, nr)
theta = np.linspace(thetal, thetau, ntheta)
phi = np.linspace(phil, phiu, nphi)

# convert degrees to radians
theta = np.deg2rad(theta)
phi = np.deg2rad(phi)

# function for cross-approximation
dims = (nqx,nqy,nl,nr,ntheta,nphi)
G_func = lambda inds: G_cylinder(qx[inds[0]], qy[inds[1]], l[inds[2]], r[inds[3]], theta[inds[4]], phi[inds[5]], drho)

# form low-rank TT-representation
tt_approx(G_func, dims, tol=1e-5, max_rank=250, compute_true_error=True, max_error_evals=1e15)
