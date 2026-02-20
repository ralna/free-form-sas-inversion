"""
Low-rank Ellipsoid Green's function approximation
https://www.sasview.org/docs/user/models/ellipsoid.html

Parameters:
qx - scattering vector x component
qy - scattering vector y component
rp - polar radius
re - equatorial radius
theta - ellipsoid axis to beam angle
phi - ellipsoid rotation about beam
drho - difference between scattering length densities

Copyright (C) 2025 The Science and Technology Facilities Council (STFC)
"""
import numpy as np
from ffsi.models.ellipsoid import G_ellipsoid
from ffsi.tensor_train import tt_approx

# contrast
drho = 1

# qx discretisation
qxl = -0.75
qxu = 0.75
nqx = 10

# qy discretisation
qyl = -.075
qyu = 0.75
nqy = 10

# rp discretisation
rpl = 50
rpu = 90
nrp = 10

# re discretisation
rel = 100
reu = 180
nre = 10

# theta discretisation
thetal = 5/180 * np.pi
thetau = 60/180 * np.pi
ntheta = 10

# phi discretisation
phil = 150/180 * np.pi
phiu = 240/180 * np.pi
nphi = 10

# discretise q, l, r, theta, phi
qx = np.linspace(qxl, qxu, nqx)
qy = np.linspace(qyl, qyu, nqy)
rp = np.linspace(rpl, rpu, nrp)
re = np.linspace(rel, reu, nre)
theta = np.linspace(thetal, thetau, ntheta)
phi = np.linspace(phil, phiu, nphi)

# function for cross-approximation
dims = (nqx,nqy,nrp,nre,ntheta,nphi)
G_func = lambda inds: G_ellipsoid(qx[inds[0]], qy[inds[1]], rp[inds[2]], re[inds[3]], theta[inds[4]], phi[inds[5]], drho)

# form low-rank TT-representation
tt_approx(G_func, dims, tol=1e-5, max_rank=250, compute_true_error=True, max_error_evals=1e15)
