"""
Free-form SAS Optimization Interface (Sphere version)

Mandatory Parameters:
G - Green's function
dims - dimensions of each Green's tensor index
I_data - intensity data
I_data_std - error on intensity data

Optional Parameters:
sigma - regularization parameter value

Returns:
xi_opt - optimal xi
b_opt - optimal b
w_r_opt - optimal w_r

Example usage:

xi_opt, b_opt, w_opt = tt_optimize(G, dims, I_data, I_data_std)

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import numpy as np
import cupy as cp
from galahad import snls


def tt_optimize(G, dims, I_data, I_data_std, sigma=None):

    # use CPU or GPU as appropriate
    xp = cp.get_array_module(G, dims, I_data, I_data_std)
    print("(using " + xp.__name__ + " for residual and Jacobian computation)")

    # TODO: this is very special case for sphere
    nq, nr = dims

    # w0 is uniform distribution
    w_r_0 = xp.ones(nr) / nr

    # this averages out G over the parameters
    # TODO: this is special case as the r core is the last core
    G_ave = xp.sum(G, axis=1) / nr

    # and xi0 and b0 can be determined from
    # min [1/sigma * (xi G_ave + b 1 - mu) ]^ 2
    mu_over_nv = I_data / I_data_std
    one_over_nv = 1 / I_data_std
    G_ave_over_nv = G_ave / I_data_std
    a11 = xp.sum(G_ave_over_nv ** 2)
    a12 = xp.sum(G_ave_over_nv * one_over_nv)
    a22 = xp.sum(one_over_nv ** 2)
    b1 = xp.sum(mu_over_nv * G_ave_over_nv)
    b2 = xp.sum(mu_over_nv * one_over_nv)

    # solve xi0 and b0 using Cramer's rule
    A = a11 * a22 - a12 * a12
    xi0 = (b1 * a22 - b2 * a12) / A
    b0 = (b2 * a11 - b1 * a12) / A
    print('xi0: %.2e' % xi0)
    print('b0: %.2e' % b0)

    # determine xi0 and b0 scaling
    xi_sc = 10 ** xp.floor(xp.log10(xp.abs(xi0)))
    b_sc = 10 ** xp.floor(xp.log10(xp.abs(b0)))

    # form residuals
    # TODO: this is special case as the r core is the last core
    def eval_r(x):

        # move x to GPU if required
        x = xp.asarray(x)

        # extract variable scalings
        xi = x[0] * xi_sc
        b = x[1] * b0
        w_r = x[2:] # in [0,1]

        # form Gw (the form factor)
        Gw = G @ w_r

        # intensity from forward model
        I_model = xi * Gw + b

        # intensity misfit
        eps = (I_model - I_data) / I_data_std

        # handle regularization
        if sigma is None: # no regularization
            res = eps
        else: # regularisation term sigma(w[i+1]-w[i])
            reg = sigma * xp.diff(w_r)
            res = xp.hstack((eps,reg))

        # move residual to CPU if required
        if xp.__name__ == 'cupy':
            return res.get()
        else:
            return res

    # form Jacobian
    # TODO: this is special case
    def eval_Jr(x):

        # move x to GPU if required
        x = xp.asarray(x)

        # extract variables and unscale
        xi = x[0] * xi_sc
        w_r = x[2:] # in [0,1]

        # form Gw (the form factor)
        Gw = G @ w_r

        # xi and b derivatives
        dxi = (xi_sc *  Gw ) / I_data_std # scaled
        db = b0 / I_data_std # scaled

        # w derivative
        dw = ( xi * G ) / I_data_std[:,xp.newaxis]

        # intensity misfit derivative (flattened)
        deps = xp.hstack((dxi[:,xp.newaxis],db[:,xp.newaxis],dw)).flatten()

        # handle regularization
        if sigma is None: # no regularization
            jac = deps
        else: # regularization term derivative (sparse)
            dreg1 = sigma * xp.ones(nr-1)  # w[i+1] term
            dreg2 = -sigma * xp.ones(nr-1) # -w[i] term
            jac = xp.hstack((deps,dreg1,dreg2))

        # move Jacobian to CPU if required
        if xp.__name__ == 'cupy':
            return jac.get()
        else:
            return jac

    # set GALAHAD SNLS options
    options = snls.initialize()
    options['print_level'] = 2
    options['jacobian_available'] = 2
    #options['slls_options']['print_level'] = 1
    options['slls_options']['sbls_options']['factorization'] = 1 # use Schur-complement
    options['slls_options']['sbls_options']['symmetric_linear_solver'] = 'sytr '
    options['slls_options']['sbls_options']['definite_linear_solver'] = 'potr '
    options['sllsb_options']['symmetric_linear_solver'] = 'sytr '
    options['sllsb_options']['fdc_options']['symmetric_linear_solver'] = 'sytr '
    options['sllsb_options']['cro_options']['symmetric_linear_solver'] = 'sytr '
    # stopping criteria
    options['stop_pg_relative'] = 1e-15
    options['stop_pg_absolute'] = 1e-6

    # form and scale initial optimization variable
    x0_scaled = xp.hstack((xi0/xi_sc,b0/b_sc,w_r_0))

    # move initial guess to CPU if required
    if xp.__name__ == 'cupy':
        x0_scaled = x0_scaled.get()

    # set GALAHAD SNLS Jacobian info
    Jr_type = 'coordinate'
    if sigma is None: # no regularization
        Jr_ne = nq*(2+nr)
        # flattened intensity misfit derivative
        Jr_row = np.tile(np.arange(nq),(2+nr,1)).flatten('F')
        Jr_col = np.tile(np.arange(2+nr),nq)
    else: # regularization requested
        Jr_ne = nq*(2+nr) + 2*(nr-1)
        # flattened intensity misfit derivative
        Jr_eps_row = np.tile(np.arange(nq),(2+nr,1)).flatten('F')
        Jr_eps_col = np.tile(np.arange(2+nr),nq)
        # sparse sigma(w[i+1]-w[i]) derivative
        Jr_reg1_row = np.arange(nq,nq+nr-1)
        Jr_reg1_col = np.arange(3,2+nr) # w[i+1] term
        Jr_reg2_row = np.arange(nq,nq+nr-1)
        Jr_reg2_col = np.arange(2,2+nr-1) # -w[i] term
        # combined derivative
        Jr_row = np.hstack((Jr_eps_row,Jr_reg1_row,Jr_reg2_row))
        Jr_col = np.hstack((Jr_eps_col,Jr_reg1_col,Jr_reg2_col))
    Jr_ptr_ne = 0
    Jr_ptr = None

    # set GALAHAD SNLS cohorts
    n = 2 + nr
    if sigma is None: # no regularization
        m_r = nq
    else: # regularization requested
        m_r = nq + nr-1
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
    xi_opt = x[0] * xi_sc
    b_opt = x[1] * b0
    w_r_opt = x[2:]
    print()
    print('xi*: %.2e' % xi_opt)
    print('b*: %.2e' % b_opt)
    print('r*: %.15e' % np.linalg.norm(eval_r(x)))

    # finalise GALAHAD SNLS
    snls.terminate()

    return xi_opt, b_opt, w_r_opt
