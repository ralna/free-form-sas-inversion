"""
Free-form SAS with Structure Factor Optimization Interface

Mandatory Parameters:
tt - xfac tensor train
dims - dimensions of each Green's tensor index
I_data - intensity data
I_data_std - error on intensity data
S_func - function to compute the structure factor
r_eff_0 - initial guess for the effective radius
vol_frac_0 - initial guess for the volume fraction

Returns:
xi_opt - optimal xi
b_opt - optimal b
r_eff_opt - optimal r_eff
vol_frac_opt = optimal vol_frac
w_opt - optimal w

Example usage:

dims = (nq,nr)
G_func = lambda inds: G_sphere(q[inds[0]], r[inds[1]], drho)
tt = tt_approx(G_func, dims)
S_func = lambda iq, r_eff, vol_frac: S_hard_sphere(q[iq], r_eff, vol_frac)
xi_opt, b_opt, vol_frac_opt, w_opt = tt_optimize_structure_factor(tt, dims, I_data, I_data_std, S_func, 180, 0.05)

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
"""
import numpy as np
# FIXME: currently this just uses SciPy for testing (no sum constraints)
# for testing the inversion pipeline
from scipy.optimize._numdiff import approx_derivative
from scipy.optimize import least_squares


def tt_optimize_structure_factor(tt, dims, I_data, I_data_std, S_func, r_eff_0, vol_frac_0):

    # TODO: this is very special case for a 2-tensor
    nq = dims[0]
    nr = dims[1]
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
        r_eff = x[2] # no need to scale
        vol_frac = x[3] # in [0,1]
        w = x[4:] # in [0,1]

        # form Gw (the form factor)
        Gw = core1[0,:,:] @ np.tensordot(core2[:,:,0], w, axes=(1,0))

        # form S (the structure factor)
        S = np.zeros(nq)
        for iq in range(nq):
            S[iq] = S_func(iq, r_eff, vol_frac)

        # intensity from forward model
        I_model = xi * vol_frac * Gw * S + b

        # intensity misfit
        eps = (I_model - I_data) / I_data_std

        return eps

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
    A = a11 * a22 - a12 * a12
    xi0 = (b1 * a22 - b2 * a12) / A
    b0 = (b2 * a11 - b1 * a12) / A
    print('xi0: %.2e' % xi0)
    print('b0: %.2e' % b0)

    # call SciPy least squares with variable scaling
    print('\nCalling SciPy least_squares...')
    x0_scaled = np.hstack((1,1,r_eff_0,vol_frac_0,w0))
    ub = np.ones(x0_scaled.size)
    ub[2] = np.inf # r_eff has no upper bound
    result = least_squares(res, x0_scaled, bounds=(0,ub), verbose=2, kwargs={'xi0':xi0,'b0':b0})

    # extract results and unscale
    xi_opt = result.x[0] * xi0
    b_opt = result.x[1] * b0
    r_eff_opt = result.x[2]
    vol_frac_opt = result.x[3]
    w_opt = result.x[4:]
    print()
    print('xi*: %.2e' % xi_opt)
    print('b*: %.2e' % b_opt)
    print('r_eff*: %.2e' % r_eff_opt)
    print('vol_frac*: %.2e' % vol_frac_opt)

    return xi_opt, b_opt, r_eff_opt, vol_frac_opt, w_opt
