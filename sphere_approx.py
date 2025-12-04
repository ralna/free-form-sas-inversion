"""
Low-rank Sphere Green's function approximation
https://www.sasview.org/docs/user/models/sphere.html

Parameters:
q - scattering vector
r - sphere radius
drho - difference between scattering length densities

Copyright (C) 2025 The Science and Technology Facilities Council (STFC)
"""
import numpy as np
from models.sphere import G_sphere
import tt

# for plotting
import matplotlib.pyplot as plt

# contrast
drho = 1

# q discretisation
nq = 1000
ql = 0.00145
qu = 0.09995

# r discretisation
nr = 300
rl = 0.1
ru = 1000

# discretise q and r
q = np.linspace(ql, qu, nq)
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
plt.title("Sphere Green's Function")
plt.xlabel('Singular Value Index')
plt.ylabel('Normalised Singular Value')
plt.savefig('sphere_eigs.png')

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
