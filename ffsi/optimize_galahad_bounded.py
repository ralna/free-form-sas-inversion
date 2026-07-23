"""
Free-form SAS Optimization Interface to GALAHAD
(that respects lower and upper bounds on xi and b)

Mandatory Parameters:
G - Green's function
I_data - intensity data
I_data_std - error on intensity data
xi_lb - lower bound on xi
xi_ub - upper bound on xi
b_lb - lower bound on b
b_ub - upper bound on b

Optional Parameters:
sigma - regularization parameter value

Returns:
xi_opt - optimal xi in [xi_lb,xi_ub]
b_opt - optimal b in [b_lb,b_ub]
w_opt_list - list of optimal parameters

Example usage:

xi_opt, b_opt, w_opt_list = optimize(G, I_data, I_data_std, xi_lb, xi_ub, b_lb, b_ub)

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import numpy as np
from galahad import snls

from ffsi.optimize_galahad import optimize
from ffsi.array_module import get_array_module
from ffsi.utils import contract_tensor


def optimize_bounded(G, I_data, I_data_std, xi_lb, xi_ub, b_lb, b_ub, sigma=None):

    print('The bounds on the parameters are')
    print('xi bounds: [%.2e,%.2e]' % (xi_lb,xi_ub))
    print('b bounds: [%.2e,%.2e]' % (b_lb,b_ub))

    # use CPU or GPU as appropriate
    xp = get_array_module(G, I_data, I_data_std)
    print("INFO: using " + xp.__name__ + " for residual and Jacobian computation")

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

    ## Optimize over all parameters
    xi, b, w_list = optimize(G, I_data, I_data_std, sigma=sigma)

    ## Check if xi and b are within their bounds
    if (xi_lb <= xi <= xi_ub) and (b_lb <= b <= b_ub):

        # if so return xi and b interior solution
        print('\nUnconstrained solution satisfies bounds')
        return xi, b, w_list

    ## Otherwise find remaining solutions for xi and b
    else:
        print('\nUnconstrained solution violates bounds')
        sols = []

        # xi fixed and b fixed solutions
        sols.append((xi_lb, b_lb))
        sols.append((xi_lb, b_ub))
        sols.append((xi_ub, b_lb))
        sols.append((xi_ub, b_ub))

        # xi and b interior can be determined from
        # min [1/sigma * (xi G_ave + b 1 - mu) ]^ 2
        mu_over_nv = I_data / I_data_std
        one_over_nv = 1 / I_data_std
        Gw = contract_tensor(G, w_list, skip_axes=q_axes)
        Gw_over_nv = Gw / I_data_std
        a11 = xp.sum(Gw_over_nv ** 2)
        a12 = xp.sum(Gw_over_nv * one_over_nv)
        a22 = xp.sum(one_over_nv ** 2)
        b1 = xp.sum(mu_over_nv * Gw_over_nv)
        b2 = xp.sum(mu_over_nv * one_over_nv)

        # xi fixed, b interior solutions
        # xi = xi_lb, b interior
        b_int = (b2 - a12 * xi_lb) / a22
        if b_lb <= b_int <= b_ub:
            sols.append((xi_lb, b_int))
        # xi = xi_ub, b interior
        b_int = (b2 - a12 * xi_ub) / a22
        if b_lb <= b_int <= b_ub:
            sols.append((xi_ub, b_int))

        # b fixed, xi interior solutions
        # b = b_lb, xi interior
        xi_int = (b1 - a12 * b_lb) / a11
        if xi_lb <= xi_int <= xi_ub:
            sols.append((xi_int, b_lb))
        # b = b_ub, xi interior
        xi_int = (b1 - a12 * b_ub) / a11
        if xi_lb <= xi_int <= xi_ub:
            sols.append((xi_int, b_ub))

        # determine which of the solutions is optimal
        xi_opt = 0
        b_opt = 0
        rnorm = xp.inf
        for xi,b in sols:
            I_model = xi * Gw + b
            eps = (I_model - I_data) / I_data_std
            res = xp.linalg.norm(eps)
            if  res < rnorm:
                rnorm = res
                xi_opt = xi
                b_opt = b

        ## run SNLS with the optimal xi, b fixed
        print('\nOptimal xi and b that satisfy bounds have')
        print('xi*: %.2e' % xi_opt)
        print('b*: %.2e' % b_opt)
        print('r*: %.15e' % rnorm)

        # form residuals
        def eval_r(x):

            # move x to GPU if required
            x = xp.asarray(x)

            # extract variables and unscale
            split_inds = np.cumsum(p_dims)[:-1]
            w_list = xp.split(x, split_inds)

            # form Gw (the form factor)
            Gw = contract_tensor(G, w_list, skip_axes=q_axes)

            # intensity from forward model
            I_model = xi_opt * Gw + b_opt

            # intensity misfit
            eps = (I_model - I_data) / I_data_std

            # handle regularization
            if sigma is None: # no regularization
                res = eps.reshape(-1)
            else: # regularisation terms sigma(w[i+1]-w[i])
                reg = [sigma * xp.diff(w) for w in w_list]
                res = xp.concat((eps.reshape(-1),*reg))

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
            split_inds = np.cumsum(p_dims)[:-1]
            w_list = xp.split(x, split_inds)

            # preallocate storage for intensity misfit derivative
            deps = xp.empty((*q_dims, np.sum(p_dims)))

            # w derivatives
            inds = np.cumsum((0,*p_dims))
            for i in range(len(p_dims)):
                slice_i = slice(inds[i],inds[i+1])
                w_contract_list = [w for k,w in enumerate(w_list) if k != i]
                Gw_dw = contract_tensor(G, w_contract_list, skip_axes=[*q_axes,p_axes[0]+i])
                deps[...,slice_i] = ( xi_opt * Gw_dw ) / I_data_std[...,None]

            # flatten intensity misfit derivative
            deps = deps.reshape(-1)

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
        x0 = xp.hstack(*w_list)

        # move initial guess to CPU if required
        if xp.__name__ == 'cupy':
            x0 = x0.get()

        # set GALAHAD SNLS dimensions
        n = np.sum(p_dims)
        if sigma is None: # no regularization
            m_r = np.prod(q_dims)
        else: # regularization requested
            m_r = np.prod(q_dims) + np.sum(np.array(p_dims)-1)
        m_c = len(p_dims)

        # set GALAHAD SNLS cohorts
        ch_list = [i * np.ones(n, dtype=int) for i,n in enumerate(p_dims)]
        cohort = np.hstack(*ch_list)

        # set GALAHAD SNLS Jacobian info
        if sigma is None: # no regularization
            Jr_type = 'dense'
            Jr_ne = m_r * n
            Jr_row = None
            Jr_col = None
        else: # regularization requested
            Jr_type = 'coordinate'
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
            starts = np.cumsum(p_dims) - p_dims
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
        x, y, z, r, g, x_stat = snls.solve(n, m_r, m_c, x0, eval_r, Jr_ne, eval_Jr)

        # get information
        info = snls.information()
        #print("inform:", inform)
        print(" f: %.4f" % info['obj'])
        print('** snls exit status:', info['status'])

        # extract results
        split_inds = np.cumsum(p_dims)[:-1]
        w_opt_list = np.split(x, split_inds)

        print()
        print('Optimal solution that satisfies bounds has')
        print('xi*: %.2e' % xi_opt)
        print('b*: %.2e' % b_opt)
        print('r*: %.15e' % np.linalg.norm(eval_r(x)))

        # finalise GALAHAD SNLS
        snls.terminate()

        return xi_opt, b_opt, w_opt_list
