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
"""
import numpy as np
from ffsi.models.cylinder import G_cylinder

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
l = np.linspace(ll, lu, nl)
r = np.linspace(rl, ru, nr)
theta = np.linspace(thetal, thetau, ntheta)
phi = np.linspace(phil, phiu, nphi)

# function for cross-approximation
dims = (nqx,nqy,nl,nr,ntheta,nphi)
G_func = lambda inds: G_cylinder(qx[inds[0]], qy[inds[1]], l[inds[2]], r[inds[3]], theta[inds[4]], phi[inds[5]], drho)

# form low-rank TT-representation
tol = 1e-5
max_rank = 250
print('Computing TT-representation using xfac...')
print('Tolerance: %.2e' % tol)

t0 = time.time()
param = xfacpy.TensorCI2Param()
param.reltol = tol
param.fullPiv = True
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
G = np.zeros((nqx,nqy,nl,nr,ntheta,nphi))
for iqx in range(nqx):
    for iqy in range(nqy):
        for il in range(nl):
            for ir in range(nr):
                for it in range(ntheta):
                    for ip in range(nphi):
                        G[iqx,iqy,il,ir,it,ip] = G_cylinder(qx[iqx], qy[iqy], l[il], r[ir], theta[it], phi[ip], drho)

# compute TT approximation error
abs_err = tci.trueError(max_n_eval=int(1e15))
rel_err = abs_err / np.linalg.norm(G)
print('TT-approximation relative error: %.2e' % rel_err)
