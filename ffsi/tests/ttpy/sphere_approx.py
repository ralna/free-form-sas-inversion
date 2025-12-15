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
from ffsi.models.sphere import G_sphere
import tt

# for plotting
import matplotlib.pyplot as plt

# for timing
import time

# contrast
drho = 1

# q discretisation
ql = 1e-3
qu = 1
nq = 200

# r discretisation
rl = 400
ru = 800
nr = 500

# discretise q and r
q = np.linspace(ql, qu, nq)
r = np.linspace(rl, ru, nr)

# FIXME: forming full G should not be required!
# form Green's function
print("Forming full Green's function tensor...")
G = np.zeros((nq,nr))
for iq in range(nq):
    for ir in range(nr):
        G[iq,ir] = G_sphere(q[iq], r[ir], drho)

# FIXME: this is just for plotting
# plot singular values of G
s = np.linalg.svd(G, compute_uv=False)
plt.semilogy(s/s[0])
plt.grid()
plt.title("Sphere Green's Function (Benchmark)")
plt.xlabel('Singular Value Index')
plt.ylabel('Normalised Singular Value')
plt.savefig('sphere_eigs.png')

# form low-rank TT-representation
tol = 1e-6
print('Computing TT-representation using SVD...')
print('Tolerance: %.2e' % tol)

t0 = time.time()
GTT = tt.tensor(G,tol)
t1 = time.time()
print('SVD time: %.2f s' % (t1-t0))

print(GTT)

# form full-weight array from TT-representation
fGC = GTT.full()
E = fGC-G
error = np.linalg.norm(E)/np.linalg.norm(G)
print('TT-approximation relative error: ')
print(error)
