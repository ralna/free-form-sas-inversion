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

# contrast
drho = 1

# qx, qy discretisation
nqx = 120
nqy = 120
q_side = np.logspace(-2, 0, 50) # log scale on the sides
q_center = np.linspace(-0.0095, 0.0095, 20) # linear scale in the cente
qx = np.hstack((-q_side[::-1], q_center, q_side))
qy = qx.copy()

# l discretisation
ll = 200
lu = 600
nl = 40

# r discretisation
rl = 50
ru = 90
nr = 40

# theta discretisation
thetal = 20
thetau = 75
ntheta = 40

# phi discretisation
phil = 150
phiu = 240
nphi = 40

# discretise l, r, theta, phi
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
tt_approx(G_func, dims, tol=1e-4, max_rank=250, compute_true_error=False)
