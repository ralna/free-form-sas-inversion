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
from models.ellipsoid import G_ellipsoid
import tt

# for plotting
import matplotlib.pyplot as plt

# contrast
drho = 1

# qx discretisation
qxl = -75
qxu = 75
nqx = 10

# qy discretisation
qyl = -75
qyu = 75
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

# FIXME: forming full G should not be required!
# form Green's function
print("Forming full Green's function tensor...")
# TODO: check if SASView has any vectorisation
G = np.zeros((nqx,nqy,nrp,nre,ntheta,nphi))
for iqx in range(nqx):
    for iqy in range(nqy):
        for irp in range(nrp):
            for ire in range(nre):
                for it in range(ntheta):
                    for ip in range(nphi):
                        G[iqx,iqy,irp,ire,it,ip] = G_ellipsoid(qx[iqx], qy[iqy], rp[irp], re[ire], theta[it], phi[ip], drho)

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
