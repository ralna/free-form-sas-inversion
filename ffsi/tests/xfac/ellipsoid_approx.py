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

# import locally compiled xfac python module
import sys
sys.path.append("../xfac/build/python")
import xfacpy

# for timing
import time

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
tol = 1e-6
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

# # FIXME: the below is for error estimation only
print('Forming G for TT-approximation error...')

# form Green's function tensor
G = np.zeros((nqx,nqy,nrp,nre,ntheta,nphi))
for iqx in range(nqx):
    for iqy in range(nqy):
        for irp in range(nrp):
            for ire in range(nre):
                for it in range(ntheta):
                    for ip in range(nphi):
                        G[iqx,iqy,irp,ire,it,ip] = G_ellipsoid(qx[iqx], qy[iqy], rp[irp], re[ire], theta[it], phi[ip], drho)

# compute TT approximation error
abs_err = tci.trueError(max_n_eval=int(1e15))
rel_err = abs_err / np.linalg.norm(G)
print('TT-approximation relative error: %.2e' % rel_err)
