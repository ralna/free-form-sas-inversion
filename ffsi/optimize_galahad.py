"""
Free-form SAS Optimization Interface to GALAHAD

Mandatory Parameters:
G - Green's function
I_data - intensity data
I_data_std - error on intensity data

Optional Parameters:
sigma - regularization parameter value

Returns:
xi_opt - optimal xi
b_opt - optimal b
w_opt_list - list of optimal parameters

Example usage:

xi_opt, b_opt, w_opt_list = optimize(G, I_data, I_data_std)

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import numpy as np
import cupy as cp
from galahad import snls

from ffsi.utils import contract_tensor

# TODO: handle both 1D and 2D intensity data
def optimize(G, I_data, I_data_std, sigma=None):

    # use CPU or GPU as appropriate
    xp = cp.get_array_module(G, I_data, I_data_std)
    print("(using " + xp.__name__ + " for residual and Jacobian computation)")

    # determine if data is 1D or 2D
    if len(I_data.shape) == 1:
        q_axes = (0,)
        p_axes = tuple(range(1,G.ndim))
        q_dims = G.shape[:1]
        p_dims = G.shape[1:]
    else:
        q_axes = (0,1)
        p_axes = tuple(range(2,G.ndim))
        q_dims = G.shape[:2]
        p_dims = G.shape[2:]

    # w0 are uniform distributions
    w0_list = [xp.ones(n) / n for n in p_dims]

    # this averages out G over the parameters
    # TODO: this is special case
    G_ave = xp.sum(G, axis=p_axes) / np.prod(p_dims)

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
    def eval_r(x):

        # move x to GPU if required
        x = xp.asarray(x)

        # extract variables and unscale
        xi = x[0] * xi_sc
        b = x[1] * b_sc
        split_inds = np.cumsum(p_dims)[:-1]
        w_list = xp.split(x[2:], split_inds)

        # form Gw (the form factor)
        Gw = contract_tensor(G, w_list, skip_axes=q_axes)

        # intensity from forward model
        I_model = xi * Gw + b

        # intensity misfit
        eps = (I_model - I_data) / I_data_std

        # handle regularization
        if sigma is None: # no regularization
            res = eps.flatten()
        else: # regularisation terms sigma(w[i+1]-w[i])
            reg = [sigma * xp.diff(w) for w in w_list]
            res = xp.concat((eps.flatten(),*reg))

        # move residual to CPU if required
        if xp.__name__ == 'cupy':
            return res.get()
        else:
            return res

    # form Jacobian
    def eval_Jr(x):

        # move x to GPU if required
        x = xp.asarray(x)

        # extract variables and unscale
        xi = x[0] * xi_sc
        split_inds = np.cumsum(p_dims)[:-1]
        w_list = xp.split(x[2:], split_inds)

        # form Gw (the form factor)
        Gw = contract_tensor(G, w_list, skip_axes=q_axes)

        # xi derivative
        dxi = (xi_sc *  Gw ) / I_data_std # scaled
        dxi = dxi.flatten() # flatten in q

        # b derivative
        db = b_sc / I_data_std # scaled
        db = db.flatten() # flatten in q

        # w derivatives
        dw_list = []
        for i in range(len(p_dims)):
            w_contract_list = [w for k,w in enumerate(w_list) if k != i]
            Gw_dw = contract_tensor(G, w_contract_list, skip_axes=[*q_axes,p_axes[0]+i])
            dw = ( xi * Gw_dw ) / I_data_std[...,None]
            dw = dw.reshape(-1, dw.shape[-1]) # flatten in q
            dw_list.append(dw)

        # intensity misfit derivative (flattened)
        deps = xp.hstack((dxi[:,None],db[:,None],*dw_list)).flatten()

        # handle regularization
        if sigma is None: # no regularization
            jac = deps
        else: # regularization term derivatives (sparse)
            dreg1 = [sigma * xp.ones(n-1) for n in p_dims]  # w[i+1] terms
            dreg2 = [-sigma * xp.ones(n-1) for n in p_dims] # -w[i] terms
            jac = xp.concat((deps,*dreg1,*dreg2))

        # move Jacobian to CPU if required
        if xp.__name__ == 'cupy':
            return jac.get()
        else:
            return jac

    # set GALAHAD SNLS options
    options = snls.initialize()
    options['maxit'] = 1000
    options['print_level'] = 2
    options['jacobian_available'] = 2
    #options['slls_options']['print_level'] = 1
    options['slls_options']['maxit'] = 250
    options['slls_options']['sbls_options']['factorization'] = 1 # use Schur-complement
    options['slls_options']['sbls_options']['symmetric_linear_solver'] = 'sytr '
    options['slls_options']['sbls_options']['definite_linear_solver'] = 'potr '
    options['sllsb_options']['symmetric_linear_solver'] = 'sytr '
    options['sllsb_options']['fdc_options']['symmetric_linear_solver'] = 'sytr '
    options['sllsb_options']['cro_options']['symmetric_linear_solver'] = 'sytr '
    # stopping criteria
    options['stop_pg_relative'] = 1e-15
    options['stop_pg_absolute'] = 1e-7

    # form and scale initial optimization variable
    x0_scaled = xp.hstack((xi0/xi_sc,b0/b_sc,*w0_list))

    # move initial guess to CPU if required
    if xp.__name__ == 'cupy':
        x0_scaled = x0_scaled.get()

    # set GALAHAD SNLS dimensions
    n = 2 + np.sum(p_dims)
    if sigma is None: # no regularization
        m_r = np.prod(q_dims)
    else: # regularization requested
        m_r = np.prod(q_dims) + np.sum(np.array(p_dims)-1)
    m_c = len(p_dims)

    # set GALAHAD SNLS cohorts
    ch_list = [i * np.ones(n, dtype=int) for i,n in enumerate(p_dims)]
    cohort = np.hstack(( np.array([-1,-1]), *ch_list))

    # set GALAHAD SNLS Jacobian info
    Jr_type = 'coordinate'
    if sigma is None: # no regularization
        # FIXME: this should probably be 'dense' for performance reasons
        Jr_ne = m_r * n
        # flattened intensity misfit derivative
        Jr_row = np.tile(np.arange(m_r),(n,1)).flatten('F')
        Jr_col = np.tile(np.arange(n),m_r)
    else: # regularization requested
        # FIXME: this needs to stay as 'coordinate'
        nq = np.prod(q_dims)
        Jr_ne = nq*n + 2*np.sum(np.array(p_dims)-1)
        # flattened intensity misfit derivative
        Jr_eps_row = np.tile(np.arange(nq),(n,1)).flatten('F')
        Jr_eps_col = np.tile(np.arange(n),nq)
        # sparse regularization derivatives for w
        split_inds = np.cumsum(np.array(p_dims)-1)[:-1]
        Jr_reg1_row = np.split(np.arange(nq,nq+np.sum(np.array(p_dims)-1)), split_inds)
        Jr_reg2_row = Jr_reg1_row.copy()
        Jr_reg1_col = [] # w[i+1] terms
        Jr_reg2_col = [] # -w[i] terms
        starts = np.cumsum(p_dims) + 2 - p_dims
        for st, dim in zip(starts, p_dims):
            Jr_reg1_col.append(np.arange(st+1, st+dim))
            Jr_reg2_col.append(np.arange(st, st+dim-1))
        # combined derivative
        Jr_row = np.concat((Jr_eps_row,*Jr_reg1_row,*Jr_reg2_row))
        Jr_col = np.concat((Jr_eps_col,*Jr_reg1_col,*Jr_reg2_col))
    Jr_ptr_ne = 0
    Jr_ptr = None

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
    b_opt = x[1] * b_sc
    split_inds = np.cumsum(p_dims)[:-1]
    w_opt_list = np.split(x[2:], split_inds)

    print()
    print('xi*: %.2e' % xi_opt)
    print('b*: %.2e' % b_opt)
    print('r*: %.15e' % np.linalg.norm(eval_r(x)))

    # finalise GALAHAD SNLS
    snls.terminate()

    return xi_opt, b_opt, w_opt_list
