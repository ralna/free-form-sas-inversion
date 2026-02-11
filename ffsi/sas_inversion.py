"""
Free-form SAS Inversion Script

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
"""
import numpy as np
from ffsi.models.sphere import G_sphere

# import locally compiled xfac python module
import sys
sys.path.append("../xfac/build/python")
import xfacpy

# for timing and plotting
import time
import matplotlib.pyplot as plt

# for testing the inversion pipeline
from scipy.optimize._numdiff import approx_derivative
from scipy.optimize import least_squares

## Step 1: Approximate Green's function
print("Step 1: Approximate Green's function\n")

# contrast
drho = 1

# q discretisation (log)
ql = -3
qu = 0
nq = 200

# r discretisation
rl = 400
ru = 800
nr = 500

# discretise q and r
q = np.logspace(ql, qu, nq)
r = np.linspace(rl, ru, nr)

# function for cross-interpolation
dims = (nq,nr)
G_func = lambda inds: G_sphere(q[inds[0]], r[inds[1]], drho)

# form low-rank TT-representation
tol = 1e-10
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

# FIXME: the below is for error estimation only
print('Forming G for TT-approximation error...')

# form Green's function tensor
G = np.zeros((nq,nr))
for iq in range(nq):
    for ir in range(nr):
        G[iq,ir] = G_sphere(q[iq], r[ir], drho)

# compute TT approximation error
abs_err = tci.trueError(max_n_eval=int(1e15))
rel_err = abs_err / np.linalg.norm(G)
print('TT-approximation relative error: %.2e' % rel_err)

# FIXME: this is just for plotting
# plot singular values of G
s = np.linalg.svd(G, compute_uv=False)
plt.figure()
plt.semilogy(s/s[0])
plt.grid()
plt.title("Singular Value Decay")
plt.xlabel('Singular Value Index')
plt.ylabel('Normalised Singular Value')
plt.show()

## Step 2: Generate ground truth
print("\nStep 2: Generate ground truth\n")

# ground truth radii distributions
gaussian = np.exp(-(r - 500)**2 / (2*10**2))
boltzmann = 0.7 * np.exp(-np.abs(r - 700) / 20)
w_true = gaussian + boltzmann
w_true /= w_true.sum()  # normalize

# plot ground truth
plt.figure()
plt.plot(r, w_true)
plt.grid()
plt.title("Ground truth distributions")
plt.xlabel(r"Radius $r$ ($\AA$)")
plt.ylabel(r"Weights $w$")
plt.show()

# ground truth of scale and background
scale_true = 2
b_true = 0.5
print('b_true:', b_true)

# compute the ground truth of xi
V = 4/3 * np.pi * r ** 3 # sphere volume
V_ave = V @ w_true
xi_true = 1e-4 * scale_true / V_ave
print('xi_true:', xi_true)

# compute intensities
I_data = xi_true * G @ w_true + b_true

# add a 20%~30% error bar
np.random.seed(0)
I_data_std = (np.random.rand(len(q)) * 0.1 + 0.2) * I_data

# plot intensities
plt.figure()
plt.errorbar(q, I_data, yerr=I_data_std, ecolor='gray')
plt.grid()
plt.xscale('log')
plt.yscale('log')
plt.title('Intensity')
plt.xlabel(r"Scattering vector $q$ ($\AA^{-1}$)")
plt.ylabel(r"Intensity $I$ ($\mathrm{cm}^{-1}$)")
plt.show()

## Step 3: SAS Inversion with low-rank G
print("\nStep 3: SAS inversion with low-rank G\n")
core1 = tci.tt.core[0]
core2 = tci.tt.core[1]

# form residuals
# TODO: this is special case as the r core is the last core
def res(x, *args, **kwargs):

    # extract variables
    xi = x[0]
    b = x[1]
    w = x[2:]

    # form Gw
    Gw = core1[0,:,:] @ np.tensordot(core2[:,:,0], w, axes=(1,0))

    # intensity from forward model
    I_model = xi * Gw + b

    # intensity misfit
    eps = (I_model - I_data) / I_data_std

    return eps

# check residual
x_true = np.hstack((xi_true,b_true,w_true))
eps = np.abs(res(x_true))
print('Residual value (min,mean,max): %.2e %.2e %.2e' % (np.min(eps),np.mean(eps),np.max(eps)))

# form Jacobian
# TODO: this is special case as the r core is the last core
def jac(x, *args, **kwargs):

    # extract variables
    xi = x[0]
    b = x[1]
    w = x[2:]

    # form Gw
    Gw = core1[0,:,:] @ np.tensordot(core2[:,:,0], w, axes=(1,0))

    # xi and b derivatives
    dxi = Gw / I_data_std
    db = 1 / I_data_std

    # form G
    G = core1[0,:,:] @ core2[:,:,0]

    # w derivative
    dw = ( xi* G ) / I_data_std[:,np.newaxis]

    # intensity misfit derivative
    deps = np.hstack((dxi[:,np.newaxis],db[:,np.newaxis],dw))

    return deps

# check derivative
jac1 = jac(x_true)

# numdiff derivative
jac2 = approx_derivative(res, x_true)

# Jacobian difference
ej = np.abs(jac1-jac2)
print('Jacobian difference (min,mean,max): %.2e %.2e %.2e' % (np.min(ej),np.mean(ej),np.max(ej)))

# determinine xi0 and b0
# compute mean of G
# TODO: this is special case as the r core is the last core
G_ave = (core1[0,:] @ np.sum(core2[:,:,0], axis=1)) / nr

# solve for xi0 and b0
# min [1/sigma * (xi G_ave + b 1 - mu) ]^ 2
mu_over_nv = I_data / I_data_std
one_over_nv = 1 / I_data_std
G_ave_over_nv = G_ave / I_data_std
a11 = G_ave_over_nv @ G_ave_over_nv
a12 = G_ave_over_nv @ one_over_nv
a22 = one_over_nv @ one_over_nv
b1 = mu_over_nv @ G_ave_over_nv
b2 = mu_over_nv @ one_over_nv

# FFSAS code: Cramer's rule
print("\nCramer's Rule")
A = a11 * a22 - a12 * a12
xi0 = (b1 * a22 - b2 * a12) / A
b0 = (b2 * a11 - b1 * a12) / A
print('xi0:', xi0)
print('b0:', b0)

# LAPACK LU Decomposition
print("LAPACK DGESV")
A = np.array([[a11, a12],[a12,a22]])
b = np.array([b1,b2])
x0 = np.linalg.solve(A,b)
print('xi0:', x0[0])
print('b0:', x0[1])

# w0 is uniform distribution
w0 = np.ones(nr) / nr

# call SciPy least squares
x0 = np.hstack((xi0,b0,w0))
#info = least_squares(res, x0, )
