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
from models.cylinder import G_cylinder
import tt

# for plotting
import matplotlib.pyplot as plt

# contrast
drho = 1

# qx discretisation
nqx = 10
qxl = 0.00145
qxu = 0.09995

# qy discretisation
nqy = 10
qyl = 0.00145
qyu = 0.09995

# l discretisation
nl = 10
ll = 500
lu = 1000

# r discretisation
nr = 10
rl = 500
ru = 1000

# theta discretisation
ntheta = 10
thetal = 30/180 * np.pi
thetau = 60/180 * np.pi

# phi discretisation
nphi = 10
phil = 10/180 * np.pi
phiu = 40/180 * np.pi

# discretise q, l, r, theta, phi
qx = np.linspace(qxl, qxu, nqx)
qy = np.linspace(qyl, qyu, nqy)
l = np.linspace(ll, lu, nl)
r = np.linspace(rl, ru, nr)
theta = np.linspace(thetal, thetau, ntheta)
phi = np.linspace(phil, phiu, nphi)

# FIXME: forming full G should not be required!
# form Green's function
# TODO: check if SASView has any vectorisation
G = np.zeros((nqx,nqy,nl,nr,ntheta,nphi))
for iqx in range(nqx):
    for iqy in range(nqy):
        for il in range(nl):
            for ir in range(nr):
                for it in range(ntheta):
                    for ip in range(nphi):
                        G[iqx,iqy,il,ir,it,ip] = G_cylinder(qx[iqx], qy[iqy], l[il], r[ir], theta[it], phi[ip], drho)

# form low-rank TT-representation
tol = 1e-4
GTT = tt.tensor(G,tol)
print(GTT)

# form full-weight array from TT-representation
fGC = GTT.full()
E = fGC-G
error = np.linalg.norm(E)/np.linalg.norm(G)
print('TT-approximation relative error: ')
print(error)
