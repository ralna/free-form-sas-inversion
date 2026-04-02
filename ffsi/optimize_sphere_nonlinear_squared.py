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
import scipy.optimize as opt
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

    # determine xi0 and b0 scaling
    xi_sc = 10 ** np.floor(np.log10(np.abs(xi0)))
    b_sc = 10 ** np.floor(np.log10(np.abs(b0)))

    # form residuals
    # TODO: this is special case as the r core is the last core
    def res(x):

        # extract variables and unscale
        xi = x[0] * xi_sc
        b = x[1] * b_sc
        s_r = x[2:] # in [0,1]

        # form Gs2 (the form factor)
        Gs2 = core_q[0,:,:] @ np.tensordot(core_r[:,:,0], s_r ** 2, axes=(1,0))

        # intensity from forward model
        I_model = xi * Gs2 + b

        # intensity misfit
        eps = (I_model - I_data) / I_data_std

        return eps

    # form Jacobian
    # TODO: this is special case as the r core is the last core
    def jac(x):

        # extract variables and unscale
        xi = x[0] * xi_sc
        s_r = x[2:] # in [0,1]

        # form Gs2 (the form factor)
        Gs2 = core_q[0,:,:] @ np.tensordot(core_r[:,:,0], s_r ** 2, axes=(1,0))

        # xi and b derivatives
        dxi = (xi_sc *  Gs2 ) / I_data_std # scaled
        db = b_sc / I_data_std # scaled

        # form G
        G = core_q[0,:,:] @ core_r[:,:,0]

        # s derivative
        ds = ( 2 * xi * G * s_r ) / I_data_std[:,np.newaxis] # scaled

        # intensity misfit derivative
        deps = np.hstack((dxi[:,np.newaxis],db[:,np.newaxis],ds))

        return deps

    # form second Hessian term
    # TODO: this is very special case
    def hess_term2(x):

        # extract variables and unscale
        xi = x[0] * xi_sc
        s_r = x[2:] # in [0,1]

        # form G
        G = core_q[0,:,:] @ core_r[:,:,0]

        # form off-diagonal nonzero block
        dsxi = ( 2 * xi_sc * G * s_r ) / I_data_std[:,np.newaxis] # scaled

        # form diagonal block
        ds2 = ( 2 * xi * G ) / I_data_std[:,np.newaxis]

        # form residual
        r = res(x)

        # form the two nonzero blocks
        r_ds2 = np.sum(r[:,np.newaxis] * ds2, axis=0)
        r_dsxi = np.sum(r[:,np.newaxis] * dsxi, axis=0)

        # form the Hessian term and insert blocks
        ht2 = np.zeros((2+nr,2+nr))
        ht2[0,2:] = r_dsxi
        ht2[2:,0] = r_dsxi
        ht2[2:,2:] = r_ds2 * np.eye(nr)

        return ht2

    # check residual
    if check_residual:
        x_true_scaled = np.hstack((xi_true/xi_sc, b_true/b_sc, np.sqrt(w_r_true)))
        eps = np.abs(res(x_true_scaled))
        print('\nResidual value (min,mean,max): %.2e %.2e %.2e' % (np.min(eps),np.mean(eps),np.max(eps)))

    # check derivative
    if check_derivative:
        jac1 = jac(x_true_scaled)
        jac2 = approx_derivative(res, x_true_scaled) # numdiff derivative
        ej = np.abs(jac1-jac2)
        print('\nJacobian difference (min,mean,max): %.2e %.2e %.2e' % (np.min(ej),np.mean(ej),np.max(ej)))

    # form simplex constraint
    # TODO: this is special case of one constraint
    def con(x):

        # extract variables
        s_r = x[2:] # in [0,1]

        # simplex constraint
        s = np.sum(s_r ** 2) - 1

        return s

    # form simplex gradient
    # TODO: this is special case of one constraint
    def congrad(x):

        # extract variables
        s_r = x[2:] # in [0,1]

        # simplex gradient
        sjac = np.zeros(2+nr)
        sjac[0] = 0
        sjac[1] = 0
        sjac[2:] = 2 * s_r

        return sjac

    # form simplex Hessian-vector product
    # TODO: this is special case of one constraint
    def conhessprod(x,v):
        H = np.zeros((2+nr,2+nr))
        H[2:,2:] = 2 * v * np.eye(nr)
        return H

    # check constraint
    if check_residual:
        eps = np.abs(con(x_true_scaled))
        print('\nConstraint value (min,mean,max): %.2e %.2e %.2e' % (np.min(eps),np.mean(eps),np.max(eps)))

    # check constraint gradient
    if check_derivative:
        grad1 = congrad(x_true_scaled)
        grad2 = approx_derivative(con, x_true_scaled) # numdiff derivative
        eg = np.abs(grad1-grad2)
        print('\nConstraint gradient difference (min,mean,max): %.2e %.2e %.2e' % (np.min(eg),np.mean(eg),np.max(eg)))

    # check constraint Hessian
    if check_derivative:
        hess1 = conhessprod(x_true_scaled,1)
        hess2 = approx_derivative(congrad, x_true_scaled) # numdiff derivative
        eh = np.abs(hess1-hess2)
        print('\nConstrain Hessian difference (min,mean,max): %.2e %.2e %.2e' % (np.min(eh),np.mean(eh),np.max(eh)))

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
        hess1 = objhess(x_true_scaled)
        hess2 = approx_derivative(objgrad, x_true_scaled) # numdiff derivative
        eh = np.abs(hess1-hess2)
        print('\nHessian difference (min,mean,max): %.2e %.2e %.2e' % (np.min(eh),np.mean(eh),np.max(eh)))

    # setup simplex equality constraint
    # TODO: this is special case
    objcon = opt.NonlinearConstraint(con, 0, 0, jac=congrad, hess=conhessprod)

    # call SciPy minimize with variable scaling
    print('\nCalling SciPy minimize...')
    s0_scaled = np.hstack((xi0/xi_sc,b0/b_sc,np.sqrt(w_r_0)))
    result = opt.minimize(objfun, s0_scaled, jac=objgrad, hess=objhess, constraints=objcon, method='trust-constr', options={'verbose':3,'maxiter':200})

    # extract results and unscale
    xi_opt = result.x[0] * xi_sc
    b_opt = result.x[1] * b_sc
    w_r_opt = result.x[2:] ** 2
    print()
    print('xi*: %.2e' % xi_opt)
    print('b*: %.2e' % b_opt)
    print('r*: %.15e' % np.sqrt(2*objfun(result.x)))

    return xi_opt, b_opt, w_r_opt
