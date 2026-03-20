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
import scipy.optimize as opt
# for testing the inversion pipeline
from scipy.optimize._numdiff import approx_derivative


# TODO: handle both 1D and 2D intensity data
def tt_optimize(tt, dims, I_data, I_data_std,
                check_residual=False, check_derivative=False, xi_true=None, b_true=None,
                w_l_true=None, w_r_true=None, w_theta_true=None, w_phi_true=None):

    # TODO: this is very special case for cylinder
    nqx, nqy, nl, nr, ntheta, nphi = dims
    core_qx, core_qy, core_l, core_r, core_theta, core_phi = tt.core

    # indices


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
    def res(x):

        # extract variables and unscale
        xi = x[0] * xi0
        b = x[1] * b0
        s_l = x[2:2+nl] # in [0,1]
        s_r = x[2+nl:2+nl+nr] # in [0,1]
        s_theta = x[2+nl+nr:2+nl+nr+ntheta] # in [0,1]
        s_phi = x[2+nl+nr+ntheta:] # in [0,1]

        # form Gs2 (the form factor)
        Gs2 = np.tensordot(core_qx[0,:,:], core_qy, axes=(1,0)) \
            @ np.tensordot(core_l, s_l ** 2, axes=(1,0)) \
            @ np.tensordot(core_r, s_r ** 2, axes=(1,0)) \
            @ np.tensordot(core_theta, s_theta ** 2, axes=(1,0)) \
            @ np.tensordot(core_phi[:,:,0], s_phi ** 2, axes=(1,0))

        # intensity from forward model
        I_model = xi * Gs2 + b

        # intensity misfit
        eps = (I_model - I_data) / I_data_std

        return eps.flatten()

    # form Jacobian
    # TODO: this is special case
    def jac(x):

        # extract variables and unscale
        xi = x[0] * xi0
        s_l = x[2:2+nl] # in [0,1]
        s_r = x[2+nl:2+nl+nr] # in [0,1]
        s_theta = x[2+nl+nr:2+nl+nr+ntheta] # in [0,1]
        s_phi = x[2+nl+nr+ntheta:] # in [0,1]

        # form Gq matrix and compress parameter cores
        Gq = np.tensordot(core_qx[0,:,:], core_qy, axes=(1,0))
        Gs2_l = np.tensordot(core_l, s_l ** 2, axes=(1,0))
        Gs2_r = np.tensordot(core_r, s_r ** 2, axes=(1,0))
        Gs2_theta = np.tensordot(core_theta, s_theta ** 2, axes=(1,0))
        Gs2_phi = np.tensordot(core_phi[:,:,0], s_phi ** 2, axes=(1,0))

        # form derivative cores
        core_2sl = core_l * 2 * s_l[:,np.newaxis]
        core_2sr = core_r * 2 * s_r[:,np.newaxis]
        core_2st = core_theta * 2 * s_theta[:,np.newaxis]
        core_2sp = core_phi[:,:,0] * 2 * s_phi

        # form Gs2 (the form factor)
        Gs2 = Gq @ Gs2_l @ Gs2_r @ Gs2_theta @ Gs2_phi

        # xi and b derivatives
        dxi = (xi0 *  Gs2 ) / I_data_std # scaled
        db = b0 / I_data_std # scaled

        # s_l derivative
        Gs_dsl = np.tensordot(Gq, core_2sl, axes=(-1,0)) @ Gs2_r @ Gs2_theta @ Gs2_phi
        dsl = ( xi * Gs_dsl ) / I_data_std[:,:,np.newaxis]

        # s_r derivative
        # TODO: cleaner nested tensor product syntax
        Gs_dsr = np.tensordot(Gq, np.tensordot(Gs2_l, core_2sr, axes=(-1,0)), axes=(-1,0)) @ Gs2_theta @ Gs2_phi
        dsr = ( xi * Gs_dsr ) / I_data_std[:,:,np.newaxis]

        # s_theta derivative
        # TODO: much cleaner nested tensor product syntax
        Gs_dst = np.tensordot(Gq, np.tensordot(Gs2_l, np.tensordot(Gs2_r, core_2st, axes=(-1,0)), axes=(-1,0)), axes=(-1,0)) @ Gs2_phi
        dst = ( xi * Gs_dst ) / I_data_std[:,:,np.newaxis]

        # s_phi derivative
        Gs_dsp = Gq @ Gs2_l @ Gs2_r @ Gs2_theta @ core_2sp
        dsp = ( xi * Gs_dsp ) / I_data_std[:,:,np.newaxis]

        # flatten arrays in q
        dxi = dxi.flatten()
        db = db.flatten()
        dsl = dsl.reshape(-1, dsl.shape[-1])
        dsr = dsr.reshape(-1, dsr.shape[-1])
        dst = dst.reshape(-1, dst.shape[-1])
        dsp = dsp.reshape(-1, dsp.shape[-1])

        # intensity misfit derivative
        deps = np.hstack((dxi[:,np.newaxis],db[:,np.newaxis],dsl,dsr,dst,dsp))

        return deps

    # form second Hessian term
    # TODO: this is very special case
    def hess_term2(x):

        # extract variables and unscale
        xi = x[0] * xi0
        s_l = x[2:2+nl] # in [0,1]
        s_r = x[2+nl:2+nl+nr] # in [0,1]
        s_theta = x[2+nl+nr:2+nl+nr+ntheta] # in [0,1]
        s_phi = x[2+nl+nr+ntheta:] # in [0,1]

        # form residual
        r = res(x)

        # form Gq matrix and compress parameter cores
        Gq = np.tensordot(core_qx[0,:,:], core_qy, axes=(1,0))
        Gs2_l = np.tensordot(core_l, s_l ** 2, axes=(1,0))
        Gs2_r = np.tensordot(core_r, s_r ** 2, axes=(1,0))
        Gs2_theta = np.tensordot(core_theta, s_theta ** 2, axes=(1,0))
        Gs2_phi = np.tensordot(core_phi[:,:,0], s_phi ** 2, axes=(1,0))

        # form derivative cores
        core_2sl = core_l * 2 * s_l[:,np.newaxis]
        core_2sr = core_r * 2 * s_r[:,np.newaxis]
        core_2st = core_theta * 2 * s_theta[:,np.newaxis]
        core_2sp = core_phi[:,:,0] * 2 * s_phi

        # form the four nonzero xi blocks
        Gsl_xi = np.tensordot(Gq, core_2sl, axes=(-1,0)) @ Gs2_r @ Gs2_theta @ Gs2_phi
        dsl_xi = (xi0 * Gsl_xi ) / I_data_std[:,:,np.newaxis]

        # TODO: cleaner nested tensor product syntax
        Gsr_xi = np.tensordot(Gq, np.tensordot(Gs2_l, core_2sr, axes=(-1,0)), axes=(-1,0)) @ Gs2_theta @ Gs2_phi
        dsr_xi = (xi0 * Gsr_xi ) / I_data_std[:,:,np.newaxis]

        # TODO: much cleaner nested tensor product syntax
        Gst_xi = np.tensordot(Gq, np.tensordot(Gs2_l, np.tensordot(Gs2_r, core_2st, axes=(-1,0)), axes=(-1,0)), axes=(-1,0)) @ Gs2_phi
        dst_xi = (xi0 * Gst_xi ) / I_data_std[:,:,np.newaxis]

        Gsp_xi = Gq @ Gs2_l @ Gs2_r @ Gs2_theta @ core_2sp
        dsp_xi = (xi0 * Gsp_xi ) / I_data_std[:,:,np.newaxis]

        # flatten arrays in q
        dsl_xi = dsl_xi.reshape(-1, dsl_xi.shape[-1])
        dsr_xi = dsr_xi.reshape(-1, dsr_xi.shape[-1])
        dst_xi = dst_xi.reshape(-1, dst_xi.shape[-1])
        dsp_xi = dsp_xi.reshape(-1, dsp_xi.shape[-1])

        # form the nonzero blocks with r
        r_dsl_xi = np.sum(r[:,np.newaxis] * dsl_xi, axis=0)
        r_dsr_xi = np.sum(r[:,np.newaxis] * dsr_xi, axis=0)
        r_dst_xi = np.sum(r[:,np.newaxis] * dst_xi, axis=0)
        r_dsp_xi = np.sum(r[:,np.newaxis] * dsp_xi, axis=0)

        # form the six nonzero off-diagonal blocks
        # TODO: much cleaner nested tensor product syntax
        Gsl_sr = np.tensordot(Gq, np.tensordot(core_2sl, core_2sr, axes=(-1,0)), axes=(-1,0)) @ Gs2_theta @ Gs2_phi
        dsl_sr = (xi * Gsl_sr ) / I_data_std[:,:,np.newaxis,np.newaxis]

        # TODO: much cleaner nested tensor product syntax
        Gsl_st = np.tensordot(Gq, np.tensordot(core_2sl, np.tensordot(Gs2_r, core_2st, axes=(-1,0)), axes=(-1,0)), axes=(-1,0)) @ Gs2_phi
        dsl_st = (xi * Gsl_st ) / I_data_std[:,:,np.newaxis,np.newaxis]

        # TODO: much cleaner nested tensor product syntax
        Gsl_sp = np.tensordot(Gq, core_2sl, axes=(-1,0)) @ Gs2_r @ Gs2_theta @ core_2sp
        dsl_sp = (xi * Gsl_sp ) / I_data_std[:,:,np.newaxis,np.newaxis]

        # TODO: much cleaner nested tensor product syntax
        Gsr_st = np.tensordot(Gq, np.tensordot(Gs2_l, np.tensordot(core_2sr, core_2st, axes=(-1,0)), axes=(-1,0)), axes=(-1,0)) @ Gs2_phi
        dsr_st = (xi * Gsr_st ) / I_data_std[:,:,np.newaxis,np.newaxis]

        # TODO: much cleaner nested tensor product syntax
        Gsr_sp = np.tensordot(Gq, np.tensordot(Gs2_l, core_2sr, axes=(-1,0)), axes=(-1,0)) @ Gs2_theta @ core_2sp
        dsr_sp = (xi * Gsr_sp ) / I_data_std[:,:,np.newaxis,np.newaxis]

        # TODO: much cleaner nested tensor product syntax
        Gst_sp = np.tensordot(Gq, np.tensordot(Gs2_l, np.tensordot(Gs2_r, core_2st, axes=(-1,0)), axes=(-1,0)), axes=(-1,0)) @ core_2sp
        dst_sp = (xi * Gst_sp ) / I_data_std[:,:,np.newaxis,np.newaxis]

        # flatten arrays in q
        dsl_sr = dsl_sr.reshape(-1, *dsl_sr.shape[-2:])
        dsl_st = dsl_st.reshape(-1, *dsl_st.shape[-2:])
        dsl_sp = dsl_sp.reshape(-1, *dsl_sp.shape[-2:])
        dsr_st = dsr_st.reshape(-1, *dsr_st.shape[-2:])
        dsr_sp = dsr_sp.reshape(-1, *dsr_sp.shape[-2:])
        dst_sp = dst_sp.reshape(-1, *dst_sp.shape[-2:])

        # form the nonzero blocks with r
        r_dsl_sr = np.sum(r[:,np.newaxis,np.newaxis] * dsl_sr, axis=0)
        r_dsl_st = np.sum(r[:,np.newaxis,np.newaxis] * dsl_st, axis=0)
        r_dsl_sp = np.sum(r[:,np.newaxis,np.newaxis] * dsl_sp, axis=0)
        r_dsr_st = np.sum(r[:,np.newaxis,np.newaxis] * dsr_st, axis=0)
        r_dsr_sp = np.sum(r[:,np.newaxis,np.newaxis] * dsr_sp, axis=0)
        r_dst_sp = np.sum(r[:,np.newaxis,np.newaxis] * dst_sp, axis=0)

        # form the four nonzero diagonal blocks
        # TODO: really need better core multiplication syntax
        Gsl_sl = 2 * np.tensordot(Gq, core_l, axes=(-1,0)) @ Gs2_r @ Gs2_theta @ Gs2_phi
        dsl_sl = (xi * Gsl_sl ) / I_data_std[:,:,np.newaxis]

        # TODO: really need better core multiplication syntax
        Gsr_sr = 2 * np.tensordot(Gq, np.tensordot(Gs2_l, core_r, axes=(-1,0)), axes=(-1,0)) @ Gs2_theta @ Gs2_phi
        dsr_sr = (xi * Gsr_sr ) / I_data_std[:,:,np.newaxis]

        # TODO: really need better core multiplication syntax
        Gst_st = 2 * np.tensordot(Gq, np.tensordot(Gs2_l, np.tensordot(Gs2_r, core_theta, axes=(-1,0)), axes=(-1,0)), axes=(-1,0)) @ Gs2_phi
        dst_st = (xi * Gst_st ) / I_data_std[:,:,np.newaxis]

        Gsp_sp = 2 * Gq @ Gs2_l @ Gs2_r @ Gs2_theta @ core_phi[:,:,0]
        dsp_sp = (xi * Gsp_sp ) / I_data_std[:,:,np.newaxis]

        # flatten arrays in q
        dsl_sl = dsl_sl.reshape(-1, dsl_sl.shape[-1])
        dsr_sr = dsr_sr.reshape(-1, dsr_sr.shape[-1])
        dst_st = dst_st.reshape(-1, dst_st.shape[-1])
        dsp_sp = dsp_sp.reshape(-1, dsp_sp.shape[-1])

        # form the nonzero blocks with r
        r_dsl_sl = np.sum(r[:,np.newaxis] * dsl_sl, axis=0)
        r_dsr_sr = np.sum(r[:,np.newaxis] * dsr_sr, axis=0)
        r_dst_st = np.sum(r[:,np.newaxis] * dst_st, axis=0)
        r_dsp_sp = np.sum(r[:,np.newaxis] * dsp_sp, axis=0)

        # form the Hessian term and insert blocks
        # TODO: index this sensibly, this is a complete mess
        ht2 = np.zeros((2+nl+nr+ntheta+nphi,2+nl+nr+ntheta+nphi))

        # the four nonzero xi blocks (symmetric)
        ht2[0,2:2+nl] = r_dsl_xi
        ht2[0,2+nl:2+nl+nr] = r_dsr_xi
        ht2[0,2+nl+nr:2+nl+nr+ntheta] = r_dst_xi
        ht2[0,2+nl+nr+ntheta:] = r_dsp_xi
        ht2[2:2+nl,0] = r_dsl_xi
        ht2[2+nl:2+nl+nr,0] = r_dsr_xi
        ht2[2+nl+nr:2+nl+nr+ntheta,0] = r_dst_xi
        ht2[2+nl+nr+ntheta:,0] = r_dsp_xi

        # the six nonzero off-diagonal blocks (symmetric)
        ht2[2:2+nl,2+nl:2+nl+nr] = r_dsl_sr
        ht2[2:2+nl,2+nl+nr:2+nl+nr+ntheta] = r_dsl_st
        ht2[2:2+nl,2+nl+nr+ntheta:] = r_dsl_sp
        ht2[2+nl:2+nl+nr,2+nl+nr:2+nl+nr+ntheta] = r_dsr_st
        ht2[2+nl:2+nl+nr,2+nl+nr+ntheta:] = r_dsr_sp
        ht2[2+nl+nr:2+nl+nr+ntheta,2+nl+nr+ntheta:] = r_dst_sp
        ht2[2+nl:2+nl+nr,2:2+nl] = r_dsl_sr.T
        ht2[2+nl+nr:2+nl+nr+ntheta,2:2+nl] = r_dsl_st.T
        ht2[2+nl+nr+ntheta:,2:2+nl] = r_dsl_sp.T
        ht2[2+nl+nr:2+nl+nr+ntheta,2+nl:2+nl+nr] = r_dsr_st.T
        ht2[2+nl+nr+ntheta:,2+nl:2+nl+nr] = r_dsr_sp.T
        ht2[2+nl+nr+ntheta:,2+nl+nr:2+nl+nr+ntheta] = r_dst_sp.T

        # the four nonzero diagonal blocks
        ht2[2:2+nl,2:2+nl] = r_dsl_sl * np.eye(nl)
        ht2[2+nl:2+nl+nr,2+nl:2+nl+nr] = r_dsr_sr * np.eye(nr)
        ht2[2+nl+nr:2+nl+nr+ntheta,2+nl+nr:2+nl+nr+ntheta] = r_dst_st * np.eye(ntheta)
        ht2[2+nl+nr+ntheta:,2+nl+nr+ntheta:] = r_dsp_sp * np.eye(nphi)

        return ht2

     # check residual
    if check_residual:
        x_true_scaled = np.hstack((xi_true/xi0, b_true/b0, w_l_true, w_r_true, w_theta_true, w_phi_true))
        eps = np.abs(res(x_true_scaled))
        print('\nResidual value (min,mean,max): %.2e %.2e %.2e' % (np.min(eps),np.mean(eps),np.max(eps)))

    # check derivative
    if check_derivative:
        jac1 = jac(x_true_scaled)
        jac2 = approx_derivative(res, x_true_scaled) # numdiff derivative
        ej = np.abs(jac1-jac2)
        print('\nJacobian difference (min,mean,max): %.2e %.2e %.2e' % (np.min(ej),np.mean(ej),np.max(ej)))

    # form simplex constraints
    # TODO: this is special case
    def con(x):

        # extract variables
        s_l = x[2:2+nl] # in [0,1]
        s_r = x[2+nl:2+nl+nr] # in [0,1]
        s_theta = x[2+nl+nr:2+nl+nr+ntheta] # in [0,1]
        s_phi = x[2+nl+nr+ntheta:] # in [0,1]

        # simplex constraints
        s = np.hstack(( np.sum(s_l ** 2) - 1,  np.sum(s_r ** 2) - 1, np.sum(s_theta ** 2) - 1, np.sum(s_phi ** 2) - 1))

        return s

    # form simplex gradient
    # TODO: this is special case
    def congrad(x):

        # extract variables
        s_l = x[2:2+nl] # in [0,1]
        s_r = x[2+nl:2+nl+nr] # in [0,1]
        s_theta = x[2+nl+nr:2+nl+nr+ntheta] # in [0,1]
        s_phi = x[2+nl+nr+ntheta:] # in [0,1]

        # simplex gradient
        sjac = np.zeros((4,2+nl+nr+ntheta+nphi))
        sjac[0,2:2+nl] = 2 * s_l
        sjac[1,2+nl:2+nl+nr] = 2 * s_r
        sjac[2,2+nl+nr:2+nl+nr+ntheta] = 2 * s_theta
        sjac[3,2+nl+nr+ntheta:] = 2 * s_phi

        return sjac

    # form simplex Hessian-vector product
    # TODO: this is special case
    def conhessprod(x,v):
        H = np.zeros((2+nl+nr+ntheta+nphi,2+nl+nr+ntheta+nphi))
        H[2:2+nl,2:2+nl] = 2 * v[0] * np.eye(nl)
        H[2+nl:2+nl+nr,2+nl:2+nl+nr] = 2 * v[1] * np.eye(nr)
        H[2+nl+nr:2+nl+nr+ntheta,2+nl+nr:2+nl+nr+ntheta] = 2 * v[2] * np.eye(ntheta)
        H[2+nl+nr+ntheta:,2+nl+nr+ntheta:] = 2 * v[3] * np.eye(nphi)
        return H

    # check constraint gradient
    if check_derivative:
        grad1 = congrad(x_true_scaled)
        grad2 = approx_derivative(con, x_true_scaled) # numdiff derivative
        eg = np.abs(grad1-grad2)
        print('\nConstraint gradient difference (min,mean,max): %.2e %.2e %.2e' % (np.min(eg),np.mean(eg),np.max(eg)))

    # check constraint Hessian
    #if check_derivative:
    #    hess1 = conhessprod(x_true_scaled, [1,1,1,1])
    #    hess2 = approx_derivative(congrad, x_true_scaled) # numdiff derivative
    #    eh = np.abs(hess1-hess2)
    #    print('\nConstrain Hessian difference (min,mean,max): %.2e %.2e %.2e' % (np.min(eh),np.mean(eh),np.max(eh)))

    # form scalar objective
    def objfun(x):
        r = res(x)
        return 0.5 * r.T @ r

    # form scalar objective Jacobian
    def objgrad(x):
        r = res(x)
        J = jac(x)
        return J.T @ r

    # form scalar objective Hessian
    def objhess(x):
        J = jac(x)
        Ht2 = hess_term2(x)
        return J.T @ J + Ht2

    # check Hessian
    if check_derivative:
        print('\nChecking Hessian...')
        hess1 = objhess(x_true_scaled)
        hess2 = approx_derivative(objgrad, x_true_scaled) # numdiff derivative
        eh = np.abs(hess1-hess2)
        print('Hessian difference (min,mean,max): %.2e %.2e %.2e' % (np.min(eh),np.mean(eh),np.max(eh)))

    # setup simplex equality constraint
    # TODO: this is special case
    objcon = opt.NonlinearConstraint(con, 0, 0, jac=congrad, hess=conhessprod)

    # call SciPy minimize with variable scaling
    print('\nCalling SciPy minimize...')
    s0_scaled = np.hstack((xi0/xi0,b0/b0,np.sqrt(w_l_0),np.sqrt(w_r_0),np.sqrt(w_theta_0),np.sqrt(w_phi_0)))
    result = opt.minimize(objfun, s0_scaled, jac=objgrad, hess=objhess, constraints=objcon, method='trust-constr', options={'verbose':3,'maxiter':1000})

    # extract results and unscale
    xi_opt = result.x[0] * xi0
    b_opt = result.x[1] * b0
    w_l_opt = result.x[2:2+nl] ** 2
    w_r_opt = result.x[2+nl:2+nl+nr] ** 2
    w_theta_opt = result.x[2+nl+nr:2+nl+nr+ntheta] ** 2
    w_phi_opt = result.x[2+nl+nr+ntheta:] ** 2
    print()
    print('xi*: %.2e' % xi_opt)
    print('b*: %.2e' % b_opt)
    print('r*: %.15e' % np.sqrt(2*objfun(result.x)))

    return xi_opt, b_opt, w_l_opt, w_r_opt, w_theta_opt, w_phi_opt
