"""
Free-form SAS Optimization Interface (Sphere version)

Mandatory Parameters:
tt - xfac tensor train
dims - dimensions of each Green's tensor index
I_data - intensity data
I_data_std - error on intensity data

Optional Parameters:
check_residual - check residual error (slow)
check_derivative - numerically check Jacobian (slow)
xi_true - real xi for above checks
b_true - real b for above checks
w_r_true - real w_r for above checks

Returns:
xi_opt - optimal xi
b_opt - optimal b
w_r_opt - optimal w_r

Example usage:

dims = (nq,nr)
G_func = lambda inds: G_sphere(q[inds[0]], r[inds[1]], drho)
tt = tt_approx(G_func, dims)
xi_opt, b_opt, w_opt = tt_optimize(tt, dims, I_data, I_data_std)

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import numpy as np
from galahad import snls
# for testing the inversion pipeline
from scipy.optimize._numdiff import approx_derivative


def tt_optimize(tt, dims, I_data, I_data_std,
                check_residual=False, check_derivative=False, xi_true=None, b_true=None, w_r_true=None):

    # TODO: this is very special case for sphere
    nq, nr = dims
    core_q, core_r = tt.core

    # w0 is uniform distribution
    w_r_0 = np.ones(nr) / nr

    # this averages out G over the parameters
    # TODO: this is special case as the r core is the last core
    G_ave = (core_q[0,:] @ np.sum(core_r[:,:,0], axis=1)) / nr

    # and xi0 and b0 can be determined from
    # min [1/sigma * (xi G_ave + b 1 - mu) ]^ 2
    mu_over_nv = I_data / I_data_std
    one_over_nv = 1 / I_data_std
    G_ave_over_nv = G_ave / I_data_std
    a11 = np.sum(G_ave_over_nv ** 2)
    a12 = np.sum(G_ave_over_nv * one_over_nv)
    a22 = np.sum(one_over_nv ** 2)
    b1 = np.sum(mu_over_nv * G_ave_over_nv)
    b2 = np.sum(mu_over_nv * one_over_nv)

    # solve xi0 and b0 using Cramer's rule
    A = a11 * a22 - a12 * a12
    xi0 = (b1 * a22 - b2 * a12) / A
    b0 = (b2 * a11 - b1 * a12) / A
    print('xi0: %.2e' % xi0)
    print('b0: %.2e' % b0)

    # form residuals
    # TODO: this is special case as the r core is the last core
    def eval_r(x):

        # extract variable scalings
        xi = x[0] * xi0
        b = x[1] * b0
        w_r = x[2:] # in [0,1]

        # form Gw (the form factor)
        Gw = core_q[0,:,:] @ np.tensordot(core_r[:,:,0], w_r, axes=(1,0))

        # intensity from forward model
        I_model = xi * Gw + b

        # intensity misfit
        eps = (I_model - I_data) / I_data_std

        return eps

    # form Jacobian
    # TODO: this is special case as the r core is the last core
    def eval_Jr(x):

        # extract variables and unscale
        xi = x[0] * xi0
        w_r = x[2:] # in [0,1]

        # form Gw (the form factor)
        Gw = core_q[0,:,:] @ np.tensordot(core_r[:,:,0], w_r, axes=(1,0))

        # xi and b derivatives
        dxi = (xi0 *  Gw ) / I_data_std # scaled
        db = b0 / I_data_std # scaled

        # form G
        G = core_q[0,:,:] @ core_r[:,:,0]

        # w derivative
        dw = ( xi * G ) / I_data_std[:,np.newaxis]

        # intensity misfit derivative
        deps = np.hstack((dxi[:,np.newaxis],db[:,np.newaxis],dw))

        return deps.flatten()

    # check residual
    if check_residual:
        x_true_scaled = np.hstack((xi_true/xi0, b_true/b0, w_r_true))
        eps = np.abs(eval_r(x_true_scaled))
        print('\nResidual value (min,mean,max): %.2e %.2e %.2e' % (np.min(eps),np.mean(eps),np.max(eps)))

    # check derivative
    if check_derivative:
        jac1 = eval_Jr(x_true_scaled)
        jac2 = approx_derivative(eval_r, x_true_scaled) # numdiff derivative
        ej = np.abs(jac1-jac2.flatten())
        print('\nJacobian difference (min,mean,max): %.2e %.2e %.2e' % (np.min(ej),np.mean(ej),np.max(ej)))

    # set GALAHAD SNLS options
    options = snls.initialize()
    options['print_level'] = 2
    options['jacobian_available'] = 2
    #options['slls_options']['print_level'] = 1
    options['slls_options']['sbls_options']['symmetric_linear_solver'] = 'sytr '
    options['slls_options']['sbls_options']['definite_linear_solver'] = 'potr '
    options['sllsb_options']['fdc_options']['symmetric_linear_solver'] = 'sytr '
    options['sllsb_options']['cro_options']['symmetric_linear_solver'] = 'sytr '
    # stopping criteria
    options['stop_pg_relative'] = 1e-15
    options['stop_pg_absolute'] = 1e-6

    # form and scale initial optimization variable
    x0_scaled = np.hstack((1,1,w_r_0))

    # set GALAHAD SNLS Jacobian info
    Jr_type = 'dense'
    Jr_ne = nq*(2+nr)
    Jr_row = None
    Jr_col = None
    Jr_ptr_ne = 0
    Jr_ptr = None

    # set GALAHAD SNLS cohorts
    n = 2 + nr
    m_r = nq
    m_c = 1
    cohort = np.hstack((np.array([-1,-1]),np.zeros(nr, dtype=int)))

    # initialise GALAHAD SNLS
    snls.load(n, m_r, m_c, Jr_type, Jr_ne, Jr_row, Jr_col, Jr_ptr_ne, Jr_ptr, cohort, options)

    # call GALAHAD SNLS with variable scaling
    print('\nCalling GALAHAD SNLS...')
    x, y, z, r, g, x_stat = snls.solve(n, m_r, m_c, x0_scaled, eval_r, Jr_ne, eval_Jr)

    # get information
    info = snls.information()
    #print("inform:", inform)
    print(" f: %.4f" % info['obj'])
    print('** snls exit status:', info['status'])

    # extract results and unscale
    xi_opt = x[0] * xi0
    b_opt = x[1] * b0
    w_r_opt = x[2:]
    print()
    print('xi*: %.2e' % xi_opt)
    print('b*: %.2e' % b_opt)
    print('r*: %.15e' % np.linalg.norm(eval_r(x)))

    # finalise GALAHAD SNLS
    snls.terminate()

    return xi_opt, b_opt, w_r_opt
