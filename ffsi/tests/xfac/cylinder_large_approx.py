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

# qx, qy discretisation
nqx = 120
nqy = 120
q_side = np.logspace(-2, 0, 50) # log scale on the sides
q_center = np.linspace(-0.0095, 0.0095, 20) # linear scale in the cente
qx = np.hstack((-q_side, q_center, q_side))
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
thetal = 20/180 * np.pi
thetau = 75/180 * np.pi
ntheta = 40

# phi discretisation
phil = 150/180 * np.pi
phiu = 240/180 * np.pi
nphi = 40

# discretise l, r, theta, phi
l = np.linspace(ll, lu, nl)
r = np.linspace(rl, ru, nr)
theta = np.linspace(thetal, thetau, ntheta)
phi = np.linspace(phil, phiu, nphi)

# function for cross-approximation
dims = (nqx,nqy,nl,nr,ntheta,nphi)
G_func = lambda inds: G_cylinder(qx[inds[0]], qy[inds[1]], l[inds[2]], r[inds[3]], theta[inds[4]], phi[inds[5]], drho)

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
