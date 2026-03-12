"""
Free-form SAS Optimization Interface (Cylinder version)

Mandatory Parameters:
tt - xfac tensor train
dims - dimensions of each Green's tensor index
I_data - intensity data
I_data_std - error on intensity data

Returns:
xi_opt - optimal xi
b_opt - optimal b
w_l_opt - optimal w_l
w_r_opt - optimal w_r
w_theta_opt - optimal w_theta
w_phi_opt - optimal w_phi

Example usage:

dims = (nqx,nqy,nl,nr,ntheta,nphi)
G_func = lambda inds: G_cylinder(qx[inds[0]], qy[inds[1]], l[inds[2]], r[inds[3]], theta[inds[4]], phi[inds[5]], drho)
tt = tt_approx(G_func, dims)
xi_opt, b_opt, w_l_opt, w_r_opt, w_theta_opt, w_phi_opt = tt_optimize(tt, dims, I_data, I_data_std)

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import numpy as np
from galahad import snls
# for testing the inversion pipeline
from scipy.optimize._numdiff import approx_derivative


# TODO: handle both 1D and 2D intensity data
def tt_optimize(tt, dims, I_data, I_data_std, check_derivative=False):

    # TODO: this is very special case for cylinder
    nqx, nqy, nl, nr, ntheta, nphi = dims
    core_qx, core_qy, core_l, core_r, core_theta, core_phi = tt.core

    # w0 are uniform distributions
    w_l_0 = np.ones(nl) / nl
    w_r_0 = np.ones(nr) / nr
    w_theta_0 = np.ones(ntheta) / ntheta
    w_phi_0 = np.ones(nphi) / nphi

    # this averages out G over the parameters
    # TODO: this is special case
    G_ave = (np.tensordot(core_qx[0,:,:], core_qy, axes=(1,0)) \
             @ np.sum(core_l, axis=1) \
             @ np.sum(core_r, axis=1) \
             @ np.sum(core_theta, axis=1) \
             @ np.sum(core_phi[:,:,0], axis=1)) / (nl*nr*ntheta*nphi)

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
    # TODO: this is special case
    def eval_r(x):

        # extract variables and unscale
        xi = x[0] * xi0
        b = x[1] * b0
        w_l = x[2:2+nl] # in [0,1]
        w_r = x[2+nl:2+nl+nr] # in [0,1]
        w_theta = x[2+nl+nr:2+nl+nr+ntheta] # in [0,1]
        w_phi = x[2+nl+nr+ntheta:] # in [0,1]

        # form Gw (the form factor)
        Gw = np.tensordot(core_qx[0,:,:], core_qy, axes=(1,0)) \
            @ np.tensordot(core_l, w_l, axes=(1,0)) \
            @ np.tensordot(core_r, w_r, axes=(1,0)) \
            @ np.tensordot(core_theta, w_theta, axes=(1,0)) \
            @ np.tensordot(core_phi[:,:,0], w_phi, axes=(1,0))

        # intensity from forward model
        I_model = xi * Gw + b

        # intensity misfit
        eps = (I_model - I_data) / I_data_std

        return eps.flatten()

    # form Jacobian
    # TODO: this is special case
    def eval_Jr(x):

        # extract variables and unscale
        xi = x[0] * xi0
        w_l = x[2:2+nl] # in [0,1]
        w_r = x[2+nl:2+nl+nr] # in [0,1]
        w_theta = x[2+nl+nr:2+nl+nr+ntheta] # in [0,1]
        w_phi = x[2+nl+nr+ntheta:] # in [0,1]

        # form Gq matrix and compress parameter cores
        Gq = np.tensordot(core_qx[0,:,:], core_qy, axes=(1,0))
        Gw_l = np.tensordot(core_l, w_l, axes=(1,0))
        Gw_r = np.tensordot(core_r, w_r, axes=(1,0))
        Gw_theta = np.tensordot(core_theta, w_theta, axes=(1,0))
        Gw_phi = np.tensordot(core_phi[:,:,0], w_phi, axes=(1,0))

        # form Gw (the form factor)
        Gw = Gq @ Gw_l @ Gw_r @ Gw_theta @ Gw_phi

        # xi and b derivatives
        dxi = (xi0 *  Gw ) / I_data_std # scaled
        db = b0 / I_data_std # scaled

        # w_l derivative
        Gw_dwl = np.tensordot(Gq, core_l, axes=(-1,0)) @ Gw_r @ Gw_theta @ Gw_phi
        dwl = ( xi * Gw_dwl ) / I_data_std[:,:,np.newaxis]

        # w_r derivative
        # TODO: cleaner nested tensor product syntax
        Gw_dwr = np.tensordot(Gq, np.tensordot(Gw_l, core_r, axes=(-1,0)), axes=(-1,0)) @ Gw_theta @ Gw_phi
        dwr = ( xi * Gw_dwr ) / I_data_std[:,:,np.newaxis]

        # w_theta derivative
        # TODO: much cleaner nested tensor product syntax
        Gw_dwt = np.tensordot(Gq, np.tensordot(Gw_l, np.tensordot(Gw_r, core_theta, axes=(-1,0)), axes=(-1,0)), axes=(-1,0)) @ Gw_phi
        dwt = ( xi * Gw_dwt ) / I_data_std[:,:,np.newaxis]

        # w_phi derivative
        Gw_dwp = Gq @ Gw_l @ Gw_r @ Gw_theta @ core_phi[:,:,0]
        dwp = ( xi * Gw_dwp ) / I_data_std[:,:,np.newaxis]

        # flatten arrays
        dxi = dxi.flatten()
        db = db.flatten()
        dwl = dwl.reshape(-1, dwl.shape[-1])
        dwr = dwr.reshape(-1, dwr.shape[-1])
        dwt = dwt.reshape(-1, dwt.shape[-1])
        dwp = dwp.reshape(-1, dwp.shape[-1])

        # intensity misfit derivative
        deps = np.hstack((dxi[:,np.newaxis],db[:,np.newaxis],dwl,dwr,dwt,dwp))

        return deps.flatten()

    # check derivative
    if check_derivative:
        x0_scaled = np.hstack((1,1,w_l_0,w_r_0,w_theta_0,w_phi_0))
        jac1 = eval_Jr(x0_scaled)
        jac2 = approx_derivative(eval_r, x0_scaled) # numdiff derivative
        ej = np.abs(jac1-jac2.flatten())
        print('\nJacobian difference (min,mean,max): %.2e %.2e %.2e' % (np.min(ej),np.mean(ej),np.max(ej)))

    # set GALAHAD SNLS options
    options = snls.initialize()
    options['maxit'] = 1000
    options['print_level'] = 2
    options['jacobian_available'] = 2
    #options['slls_options']['print_level'] = 1
    options['slls_options']['sbls_options']['symmetric_linear_solver'] = 'sytr '
    options['slls_options']['sbls_options']['definite_linear_solver'] = 'potr '
    options['sllsb_options']['fdc_options']['symmetric_linear_solver'] = 'sytr '
    options['sllsb_options']['cro_options']['symmetric_linear_solver'] = 'sytr '
    # stopping criteria
    options['stop_pg_relative'] = 1e-15
    options['stop_pg_absolute'] = 1e-7

    # form and scale initial optimization variable
    x0_scaled = np.hstack((1,1,w_l_0,w_r_0,w_theta_0,w_phi_0))

    # set GALAHAD SNLS Jacobian info
    # FIXME: use dense interface once it is fixed
    Jr_type = 'coordinate'
    Jr_ne = nqx*nqy*(2+nl+nr+ntheta+nphi)
    Jr_row = np.tile(np.arange(nqx*nqy),(2+nl+nr+ntheta+nphi,1)).flatten('F')
    Jr_col = np.tile(np.arange(2+nl+nr+ntheta+nphi),nqx*nqy)
    Jr_ptr_ne = 0
    Jr_ptr = None

    # set GALAHAD SNLS cohorts
    n = 2 + nl + nr + ntheta + nphi
    m_r = nqx * nqy
    m_c = 4
    cohort = np.hstack(( np.array([-1,-1]), np.zeros(nl, dtype=int), np.ones(nr, dtype=int),
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
    xi_opt = x[0] * xi0
    b_opt = x[1] * b0
    w_l_opt = x[2:2+nl] # in [0,1]
    w_r_opt = x[2+nl:2+nl+nr] # in [0,1]
    w_theta_opt = x[2+nl+nr:2+nl+nr+ntheta] # in [0,1]
    w_phi_opt = x[2+nl+nr+ntheta:] # in [0,1]
    print()
    print('xi*: %.2e' % xi_opt)
    print('b*: %.2e' % b_opt)

    # finalise GALAHAD SNLS
    snls.terminate()

    return xi_opt, b_opt, w_l_opt, w_r_opt, w_theta_opt, w_phi_opt
