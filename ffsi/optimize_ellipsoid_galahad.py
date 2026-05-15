"""
Free-form SAS Optimization Interface (Ellipsoid version)

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
w_rp_opt - optimal w_rp
w_re_opt - optimal w_re
w_theta_opt - optimal w_theta
w_phi_opt - optimal w_phi

Example usage:

xi_opt, b_opt, w_rp_opt, w_re_opt, w_theta_opt, w_phi_opt = tt_optimize(G, dims, I_data, I_data_std)

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import numpy as np
import cupy as cp
from galahad import snls


# TODO: handle both 1D and 2D intensity data
def tt_optimize(G, dims, I_data, I_data_std, sigma=None):

    # use CPU or GPU as appropriate
    xp = cp.get_array_module(G, dims, I_data, I_data_std)
    print("(using " + xp.__name__ + " for residual and Jacobian computation)")

    # TODO: this is very special case for ellipsoid
    nqx, nqy, nrp, nre, ntheta, nphi = dims

    # w0 are uniform distributions
    w_rp_0 = xp.ones(nrp) / nrp
    w_re_0 = xp.ones(nre) / nre
    w_theta_0 = xp.ones(ntheta) / ntheta
    w_phi_0 = xp.ones(nphi) / nphi

    # this averages out G over the parameters
    # TODO: this is special case
    G_ave = xp.sum(G, axis=(2,3,4,5)) / (nrp*nre*ntheta*nphi)

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
    # TODO: this is special case
    def eval_r(x):

        # move x to GPU if required
        x = xp.asarray(x)

        # extract variables and unscale
        xi = x[0] * xi_sc
        b = x[1] * b_sc
        w_rp = x[2:2+nrp] # in [0,1]
        w_re = x[2+nrp:2+nrp+nre] # in [0,1]
        w_theta = x[2+nrp+nre:2+nrp+nre+ntheta] # in [0,1]
        w_phi = x[2+nrp+nre+ntheta:] # in [0,1]

        # form Gw (the form factor)
        Gw = (((G @ w_phi) @ w_theta) @ w_re) @ w_rp

        # intensity from forward model
        I_model = xi * Gw + b

        # intensity misfit
        eps = (I_model - I_data) / I_data_std

        # handle regularization
        if sigma is None: # no regularization
            res = eps.flatten()
        else: # regularisation terms sigma(w[i+1]-w[i])
            reg_rp = sigma * xp.diff(w_rp)
            reg_re = sigma * xp.diff(w_re)
            reg_theta = sigma * xp.diff(w_theta)
            reg_phi = sigma * xp.diff(w_phi)
            res = xp.hstack((eps.flatten(),reg_rp,reg_re,reg_theta,reg_phi))

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
        w_rp = x[2:2+nrp] # in [0,1]
        w_re = x[2+nrp:2+nrp+nre] # in [0,1]
        w_theta = x[2+nrp+nre:2+nrp+nre+ntheta] # in [0,1]
        w_phi = x[2+nrp+nre+ntheta:] # in [0,1]

        # form Gw (the form factor)
        Gw = (((G @ w_phi) @ w_theta) @ w_re) @ w_rp

        # xi and b derivatives
        dxi = (xi_sc *  Gw ) / I_data_std # scaled
        db = b_sc / I_data_std # scaled

        # w_rp derivative
        Gw_dwrp = xp.tensordot(xp.tensordot(xp.tensordot(G, w_re, axes=(3,0)), w_theta, axes=(3,0)), w_phi, axes=(3,0))
        dwrp = ( xi * Gw_dwrp ) / I_data_std[:,:,xp.newaxis]

        # w_re derivative
        Gw_dwre = xp.tensordot(xp.tensordot(xp.tensordot(G, w_rp, axes=(2,0)), w_theta, axes=(3,0)), w_phi, axes=(3,0))
        dwre = ( xi * Gw_dwre ) / I_data_std[:,:,xp.newaxis]

        # w_theta derivative
        Gw_dwt = xp.tensordot(xp.tensordot(xp.tensordot(G, w_rp, axes=(2,0)), w_re, axes=(2,0)), w_phi, axes=(3,0))
        dwt = ( xi * Gw_dwt ) / I_data_std[:,:,xp.newaxis]

        # w_phi derivative
        Gw_dwp = xp.tensordot(xp.tensordot(xp.tensordot(G, w_rp, axes=(2,0)), w_re, axes=(2,0)), w_theta, axes=(2,0))
        dwp = ( xi * Gw_dwp ) / I_data_std[:,:,xp.newaxis]

        # flatten arrays in q
        dxi = dxi.flatten()
        db = db.flatten()
        dwrp = dwrp.reshape(-1, dwrp.shape[-1])
        dwre = dwre.reshape(-1, dwre.shape[-1])
        dwt = dwt.reshape(-1, dwt.shape[-1])
        dwp = dwp.reshape(-1, dwp.shape[-1])

        # intensity misfit derivative (flattened)
        deps = xp.hstack((dxi[:,xp.newaxis],db[:,xp.newaxis],dwrp,dwre,dwt,dwp)).flatten()

        # handle regularization
        if sigma is None: # no regularization
            jac = deps
        else: # regularization term derivatives (sparse)
            dreg_rp1 = sigma * xp.ones(nrp-1)
            dreg_rp2 = -sigma * xp.ones(nrp-1)
            dreg_re1 = sigma * xp.ones(nre-1)
            dreg_re2 = -sigma * xp.ones(nre-1)
            dreg_t1 = sigma * xp.ones(ntheta-1)
            dreg_t2 = -sigma * xp.ones(ntheta-1)
            dreg_p1 = sigma * xp.ones(nphi-1)
            dreg_p2 = -sigma * xp.ones(nphi-1)
            jac = xp.hstack((deps,dreg_rp1,dreg_rp2,dreg_re1,dreg_re2,dreg_t1,dreg_t2,dreg_p1,dreg_p2))

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
    x0_scaled = xp.hstack((xi0/xi_sc,b0/b_sc,w_rp_0,w_re_0,w_theta_0,w_phi_0))

    # move initial guess to CPU if required
    if xp.__name__ == 'cupy':
        x0_scaled = x0_scaled.get()

    # set GALAHAD SNLS Jacobian info
    Jr_type = 'coordinate'
    if sigma is None: # no regularization
        Jr_ne = nqx*nqy*(2+nrp+nre+ntheta+nphi)
        # flattened intensity misfit derivative
        Jr_row = np.tile(np.arange(nqx*nqy),(2+nrp+nre+ntheta+nphi,1)).flatten('F')
        Jr_col = np.tile(np.arange(2+nrp+nre+ntheta+nphi),nqx*nqy)
    else: # regularization requested
        Jr_ne = nqx*nqy*(2+nrp+nre+ntheta+nphi) + 2*(nrp-1) + 2*(nre-1) + 2*(ntheta-1) + 2*(nphi-1)
        # flattened intensity misfit derivative
        Jr_eps_row = np.tile(np.arange(nqx*nqy),(2+nrp+nre+ntheta+nphi,1)).flatten('F')
        Jr_eps_col = np.tile(np.arange(2+nrp+nre+ntheta+nphi),nqx*nqy)
        # sparse regularization derivative for w_l
        Jr_reg_l1_row = np.arange(nqx*nqy,nqx*nqy+nrp-1)
        Jr_reg_l2_row = np.arange(nqx*nqy,nqx*nqy+nrp-1)
        Jr_reg_l1_col = np.arange(3,2+nrp) # w_l[i+1] term
        Jr_reg_l2_col = np.arange(2,2+nrp-1) # -w_l[i] term
        # sparse regularization derivative for w_r
        Jr_reg_r1_row = np.arange(nqx*nqy+nrp-1,nqx*nqy+nrp-1+nre-1)
        Jr_reg_r2_row = np.arange(nqx*nqy+nrp-1,nqx*nqy+nrp-1+nre-1)
        Jr_reg_r1_col = np.arange(2+nrp+1,2+nrp+nre) # w_r[i+1] term
        Jr_reg_r2_col = np.arange(2+nrp,2+nrp+nre-1) # -w_r[i] term
        # sparse regularization derivative for w_theta
        Jr_reg_t1_row = np.arange(nqx*nqy+nrp-1+nre-1,nqx*nqy+nrp-1+nre-1+ntheta-1)
        Jr_reg_t2_row = np.arange(nqx*nqy+nrp-1+nre-1,nqx*nqy+nrp-1+nre-1+ntheta-1)
        Jr_reg_t1_col = np.arange(2+nrp+nre+1,2+nrp+nre+ntheta) # w_theta[i+1] term
        Jr_reg_t2_col = np.arange(2+nrp+nre,2+nrp+nre+ntheta-1) # -w_theta[i] term
        # sparse regularization derivative for w_phi
        Jr_reg_p1_row = np.arange(nqx*nqy+nrp-1+nre-1+ntheta-1,nqx*nqy+nrp-1+nre-1+ntheta-1+nphi-1)
        Jr_reg_p2_row = np.arange(nqx*nqy+nrp-1+nre-1+ntheta-1,nqx*nqy+nrp-1+nre-1+ntheta-1+nphi-1)
        Jr_reg_p1_col = np.arange(2+nrp+nre+ntheta+1,2+nrp+nre+ntheta+nphi) # w_phi[i+1] term
        Jr_reg_p2_col = np.arange(2+nrp+nre+ntheta,2+nrp+nre+ntheta+nphi-1) # -w_phi[i] term
        # combined derivative
        Jr_row = np.hstack((Jr_eps_row,
                            Jr_reg_l1_row,Jr_reg_l2_row,
                            Jr_reg_r1_row,Jr_reg_r2_row,
                            Jr_reg_t1_row,Jr_reg_t2_row,
                            Jr_reg_p1_row,Jr_reg_p2_row))
        Jr_col = np.hstack((Jr_eps_col,
                            Jr_reg_l1_col,Jr_reg_l2_col,
                            Jr_reg_r1_col,Jr_reg_r2_col,
                            Jr_reg_t1_col,Jr_reg_t2_col,
                            Jr_reg_p1_col,Jr_reg_p2_col))
    Jr_ptr_ne = 0
    Jr_ptr = None

    # set GALAHAD SNLS cohorts
    n = 2 + nrp + nre + ntheta + nphi
    if sigma is None: # no regularization
        m_r = nqx * nqy
    else: # regularization requested
        m_r = nqx * nqy + nrp-1 + nre-1 + ntheta-1 + nphi-1
    m_c = 4
    cohort = np.hstack(( np.array([-1,-1]), np.zeros(nrp, dtype=int), np.ones(nre, dtype=int),
                        2*np.ones(ntheta, dtype=int), 3*np.ones(nphi, dtype=int) ))

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
    w_rp_opt = x[2:2+nrp] # in [0,1]
    w_re_opt = x[2+nrp:2+nrp+nre] # in [0,1]
    w_theta_opt = x[2+nrp+nre:2+nrp+nre+ntheta] # in [0,1]
    w_phi_opt = x[2+nrp+nre+ntheta:] # in [0,1]
    print()
    print('xi*: %.2e' % xi_opt)
    print('b*: %.2e' % b_opt)
    print('r*: %.15e' % np.linalg.norm(eval_r(x)))

    # finalise GALAHAD SNLS
    snls.terminate()

    return xi_opt, b_opt, w_rp_opt, w_re_opt, w_theta_opt, w_phi_opt
