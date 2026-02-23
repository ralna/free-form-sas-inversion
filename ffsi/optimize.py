"""
Free-form SAS Optimization Interface

Parameters:
tt - xfac tensor train
dims - dimensions of each Green's tensor index
I_data - intensity data
I_data_std - error on intensity data
check_residual - check residual error (slow)
check_derivative - numerically check Jacobian (slow)
xi_true - real xi for above checks
b_true - real b for above checks
w_true - real w for above checks

Returns:
xi_opt - optimal xi
b_opt - optimal b
w_opt - optimal w

Example usage:

dims = (nq,nr)
G_func = lambda inds: G_sphere(q[inds[0]], r[inds[1]], drho)
tt = tt_approx(G_func, dims)
xi_opt, b_opt, w_opt = tt_optimize(tt, dims, I_data, I_data_std)

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
"""
import numpy as np
# FIXME: currently this just uses SciPy for testing (no sum constraints)
# for testing the inversion pipeline
from scipy.optimize._numdiff import approx_derivative
from scipy.optimize import least_squares


def tt_optimize(tt, dims, I_data, I_data_std, check_residual=False, check_derivative=False, xi_true=None, b_true=None, w_true=None):

    # TODO: this is very special case for a 2-tensor
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
    if check_residual:
        x_true_scaled = np.hstack((xi_true/xi0,b_true/b0,w_true))
        eps = np.abs(res(x_true_scaled, xi0=xi0, b0=b0))
        print('Residual value (min,mean,max): %.2e %.2e %.2e' % (np.min(eps),np.mean(eps),np.max(eps)))

    # check derivative
    if check_derivative:
        jac1 = jac(x_true_scaled, xi0=xi0, b0=b0)
        jac2 = approx_derivative(res, x_true_scaled, kwargs={'xi0':xi0,'b0':b0}) # numdiff derivative
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

    return xi_opt, b_opt, w_opt
