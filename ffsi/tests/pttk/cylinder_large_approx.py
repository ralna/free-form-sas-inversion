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
from ...greedy.cross import greedy_cross

# for timing
import time

# contrast
drho = 1

# qx, qy discretisation
nqx = 40
nqy = 40
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
# TODO: check if SASView has any vectorisation
dims = (nqx,nqy,nl,nr,ntheta,nphi)
def G_func(inds):
    n = inds.shape[0]
    g_vals = np.zeros(n)
    for i in range(n):
        g_vals[i] = G_cylinder(qx[inds[i,0]], qy[inds[i,1]], l[inds[i,2]], r[inds[i,3]], theta[inds[i,4]], phi[inds[i,5]], drho)
    return g_vals

# form low-rank TT-representation
tol = 1e-4
nswp = 1000
print('Computing TT-representation using greedy cross...')
print('Tolerance: %.2e' % tol)
print('Max sweeps: %d' % nswp)

t0 = time.time()
cores = greedy_cross(dims, G_func, tol, nswp)
t1 = time.time()
print('Greedy cross time: %.2f s' % (t1-t0))

print('Number of cores: %d' % len(cores))
print('Core sizes:')
for i in range(len(cores)):
    print(cores[i].shape)
