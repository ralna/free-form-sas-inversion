"""
Free-form SAS Inversion Script

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
"""
import numpy as np
from ffsi.models.sphere import G_sphere
from ffsi.tensor_train import tt_approx

# for plotting
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
tt, abs_err = tt_approx(G_func, dims, tol=1e-10, compute_true_error=True)

# FIXME: the below is for error estimation only
print('Forming G for TT-approximation error...')

# form Green's function tensor
G = np.zeros((nq,nr))
for iq in range(nq):
    for ir in range(nr):
        G[iq,ir] = G_sphere(q[iq], r[ir], drho)

# compute TT approximation relative error
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
core1 = tt.core[0]
core2 = tt.core[1]

# form residuals
# TODO: this is special case as the r core is the last core
def res(x, *args, **kwargs):

    # extract variable scalings
    xi0 = kwargs['xi0']
    b0 = kwargs['b0']

    # extract variables and unscale
    xi = x[0] * xi0
    b = x[1] * b0
    w = x[2:] # in [0,1]

    # form Gw
    Gw = core1[0,:,:] @ np.tensordot(core2[:,:,0], w, axes=(1,0))

    # intensity from forward model
    I_model = xi * Gw + b

    # intensity misfit
    eps = (I_model - I_data) / I_data_std

    return eps

# form Jacobian
# TODO: this is special case as the r core is the last core
def jac(x, *args, **kwargs):

    # extract variable scalings
    xi0 = kwargs['xi0']

    # extract variables and unscale
    xi = x[0] * xi0
    w = x[2:] # in [0,1]

    # form Gw
    Gw = core1[0,:,:] @ np.tensordot(core2[:,:,0], w, axes=(1,0))

    # xi and b derivatives
    dxi = (xi0 *  Gw ) / I_data_std # scaled
    db = b0 / I_data_std # scaled

    # form G
    G = core1[0,:,:] @ core2[:,:,0]

    # w derivative
    dw = ( xi * G ) / I_data_std[:,np.newaxis]

    # intensity misfit derivative
    deps = np.hstack((dxi[:,np.newaxis],db[:,np.newaxis],dw))

    return deps

# w0 is uniform distribution
w0 = np.ones(nr) / nr

# this averages out G
# TODO: this is special case as the r core is the last core
G_ave = (core1[0,:] @ np.sum(core2[:,:,0], axis=1)) / nr

# and xi0 and b0 can be determined from
# min [1/sigma * (xi G_ave + b 1 - mu) ]^ 2
mu_over_nv = I_data / I_data_std
one_over_nv = 1 / I_data_std
G_ave_over_nv = G_ave / I_data_std
a11 = G_ave_over_nv @ G_ave_over_nv
a12 = G_ave_over_nv @ one_over_nv
a22 = one_over_nv @ one_over_nv
b1 = mu_over_nv @ G_ave_over_nv
b2 = mu_over_nv @ one_over_nv

# solve xi0 and b0 using Cramer's rule
print("\nCramer's Rule:")
A = a11 * a22 - a12 * a12
xi0 = (b1 * a22 - b2 * a12) / A
b0 = (b2 * a11 - b1 * a12) / A
print('xi0:', xi0)
print('b0:', b0)

# check residual
x_true_scaled = np.hstack((xi_true/xi0,b_true/b0,w_true))
eps = np.abs(res(x_true_scaled, xi0=xi0, b0=b0))
print('Residual value (min,mean,max): %.2e %.2e %.2e' % (np.min(eps),np.mean(eps),np.max(eps)))

# check derivative
jac1 = jac(x_true_scaled, xi0=xi0, b0=b0)

# numdiff derivative
jac2 = approx_derivative(res, x_true_scaled, kwargs={'xi0':xi0,'b0':b0})

# Jacobian difference
ej = np.abs(jac1-jac2)
print('Jacobian difference (min,mean,max): %.2e %.2e %.2e' % (np.min(ej),np.mean(ej),np.max(ej)))

# call SciPy least squares with variable scaling
print('Calling SciPy least_squares...')
x0_scaled = np.hstack((1,1,w0))
result = least_squares(res, x0_scaled, jac=jac, bounds=(0,1), verbose=2, kwargs={'xi0':xi0,'b0':b0})

# extract results and unscale
xi_opt = result.x[0] * xi0
b_opt = result.x[1] * b0
w_opt = result.x[2:]
print('xi*:', xi_opt)
print('b*:', b_opt)

# plot optimized distributions
plt.figure()
plt.plot(r, w_opt)
plt.grid()
plt.title("Optimized distributions")
plt.xlabel(r"Radius $r$ ($\AA$)")
plt.ylabel(r"Weights $w$")
plt.show()
