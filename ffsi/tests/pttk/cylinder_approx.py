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


# FIXME: the below is for error estimation only
print('Computing TT-approximation error...')

# form full tensor from TT approximation
fGC = np.zeros((nqx,nqy,nl,nr,ntheta,nphi))
for iqx in range(nqx):
    for iqy in range(nqy):
        for il in range(nl):
            for ir in range(nr):
                for it in range(ntheta):
                    for ip in range(nphi):
                        fGC[iqx,iqy,il,ir,it,ip] = (cores[0][:,iqx,:] @ cores[1][:,iqy,:] @ cores[2][:,il,:] @ cores[3][:,ir,:] @ cores[4][:,it,:] @ cores[5][:,ip,:])[0,0]

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
E = fGC - G
error = np.linalg.norm(E)/np.linalg.norm(G)
print('TT-approximation relative error: %.2e' % error)
