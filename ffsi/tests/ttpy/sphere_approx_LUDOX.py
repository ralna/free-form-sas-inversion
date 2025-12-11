"""
Low-rank Sphere Green's function approximation (LUDOX Data)
https://www.sasview.org/docs/user/models/sphere.html

Parameters:
q - scattering vector
r - sphere radius
drho - difference between scattering length densities

Copyright (C) 2025 The Science and Technology Facilities Council (STFC)
"""
import numpy as np
from ffsi.models.sphere import G_sphere
import tt

# for plotting
import matplotlib.pyplot as plt

# load real SAXS data
data = np.loadtxt('ffsi/data/LUDOX/S49_Ludox6_1pct.dat', skiprows=1)
q = data[:,0]
nq = len(q)

# contrast
drho = 1

# r discretisation
rl = 1
ru = 10 ** 2.5
nr = 1000
r = np.linspace(rl, ru, nr)

# FIXME: forming full G should not be required!
# form Green's function
# TODO: check if SASView has any vectorisation
G = np.zeros((nq,nr))
for iq in range(nq):
    for ir in range(nr):
        G[iq,ir] = G_sphere(q[iq], r[ir], drho)

# FIXME: this is just for plotting
# plot singular values of G
s = np.linalg.svd(G, compute_uv=False)
plt.semilogy(s/s[0])
plt.grid()
plt.title("Sphere Green's Function (LUDOX Data)")
plt.xlabel('Singular Value Index')
plt.ylabel('Normalised Singular Value')
plt.savefig('sphere_LUDOX_eigs.png')

# form low-rank TT-representation
tol = 1e-6
GTT = tt.tensor(G,tol)
print(GTT)

# form full-weight array from TT-representation
fGC = GTT.full()
E = fGC-G
error = np.linalg.norm(E)/np.linalg.norm(G)
print('TT-approximation relative error: ')
print(error)
