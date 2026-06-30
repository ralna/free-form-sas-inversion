"""
SAS sensitivity analysis

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import numpy as np

from ffsi.array_module import get_array_module

from ffsi.utils import contract_tensor


def sensitivity(G, I_data, I_data_std, xi, b, w_list, compute_uncertainty=True):

    # use CPU or GPU as appropriate
    xp = get_array_module(G, w_list)
    print("INFO: using " + xp.__name__ + " for Jacobian and Hessian computation")

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

    ### Compute Residual

    # form Gw (the form factor)
    Gw = contract_tensor(G, w_list, skip_axes=q_axes)

    # intensity from forward model
    I_model = xi * Gw + b

    # intensity misfit (residual)
    res = (I_model - I_data) / I_data_std
    res = res.reshape(-1) # flatten in q

    ### Compute Jacobian

    # preallocate storage for intensity misfit derivative
    jac = xp.empty((*q_dims, 2+np.sum(p_dims)))

    # xi derivative
    jac[...,0] = Gw / I_data_std # scaled

    # b derivative
    jac[...,1] = 1 / I_data_std # scaled

    # w derivatives
    inds = np.cumsum((2,*p_dims))
    for i in range(len(p_dims)):
        slice_i = slice(inds[i],inds[i+1])
        w_contract_list = [w for k,w in enumerate(w_list) if k != i]
        Gw_dw = contract_tensor(G, w_contract_list, skip_axes=[*q_axes,p_axes[0]+i])
        jac[...,slice_i] = ( xi * Gw_dw ) / I_data_std[...,None]

    # flatten intensity misfit derivative in q
    jac = jac.reshape(-1, jac.shape[-1])

    ### Compute Hessian 2nd derivative term

    # preallocate storage
    n = 2 + np.sum(p_dims)
    hess_term2 = xp.zeros((n,n))

    # compute nonzero Hessian 2nd term blocks
    inds = np.cumsum((2,*p_dims))
    for i in range(len(p_dims)):
        slice_i = slice(inds[i],inds[i+1])

        # the dw_dxi off-diagonal blocks (symmetric)
        w_contract_list = [w for k,w in enumerate(w_list) if k != i]
        Gw_dw = contract_tensor(G, w_contract_list, skip_axes=[*q_axes,p_axes[0]+i])
        dw_dxi = Gw_dw / I_data_std[...,None]
        dw_dxi = dw_dxi.reshape(-1, dw_dxi.shape[-1]) # flatten in q

        # form blocks with residual
        r_dw_dxi = xp.sum(res[:,None] * dw_dxi, axis=0)
        hess_term2[0, slice_i] = r_dw_dxi # upper triangular part
        hess_term2[slice_i, 0] = r_dw_dxi # lower triangular part

        # off-diagonal matrix blocks
        for j in range(len(p_dims)):
            if j <= i: # skip diagonal and lower triangle
                continue
            slice_j = slice(inds[j],inds[j+1])

            # the dwi_dwj off-diagonal blocks (symmetric)
            w_contract_list = [w for k,w in enumerate(w_list) if (k != i and k != j)]
            Gw_dwi_dwj = contract_tensor(G, w_contract_list, skip_axes=[*q_axes,p_axes[0]+i,p_axes[0]+j])
            dwi_dwj = (xi * Gw_dwi_dwj) / I_data_std[...,None,None]
            dwi_dwj = dwi_dwj.reshape(-1, *dwi_dwj.shape[-2:]) # flatten in q

            # form blocks with residual
            r_dwi_dwj = xp.sum(res[:,None,None] * dwi_dwj, axis=0)
            hess_term2[slice_i, slice_j] = r_dwi_dwj   # upper triangular part
            hess_term2[slice_j, slice_i] = r_dwi_dwj.T # lower triangular part

    ### Compute Sensitivity at x

    # compute intensity misfit gradient
    grad = xp.dot(jac.T, res)

    # compute the intensity misfit Hessian
    hess = xp.dot(jac.T, jac) + hess_term2

    # compute sensitivity
    x = xp.hstack((xi,b,*w_list))
    sens = xp.tensordot(hess, x / grad, axes=1)

    # extract sensitivity variables
    sens_xi = sens[0]
    sens_b = sens[1]
    split_inds = np.cumsum(p_dims)[:-1]
    sens_w_list = xp.split(sens[2:], split_inds)

    # report sensitivity
    print('\n== Parameter Sensitivity ==')
    print('xi: %.2e' % sens_xi)
    print('b: %.2e' % sens_b)
    for i,sens_w in enumerate(sens_w_list):
        print('w'+str(i)+':')
        with xp.printoptions(formatter={'float_kind': '{:.2e}'.format}):
            print(sens_w)

    ### Compute Posterior Uncertainty at x

    if not compute_uncertainty:

        return sens_xi, sens_b, sens_w_list

    else: # compute uncertainty

        # data covariance matrix
        Cd_inv = xp.diag(I_data_std.reshape(-1) ** (-2))

        # Taken from the book Inverse Problem Theory by Albert Tarantola
        # See equation 3.56 to estimate the posterior covariance matrix
        # based on the Jacobian of the forward model I_model = xi*Gw + b
        def compute_posterior_covariance(Jf):
            JCJ = xp.tensordot(xp.tensordot(Jf.T, Cd_inv, axes=1), Jf, axes=1)
            if len(JCJ.shape) == 0:
                JCJ_inv_diag = 1 / JCJ
            else: # sparse approximate inverse
                JCJ_inv_diag = xp.diag(JCJ) / (JCJ * JCJ).sum(axis=0)
            return xp.sqrt(JCJ_inv_diag)

        # posterior uncertainty in xi
        Jf_xi = Gw.reshape(-1) # flatten in q
        std_xi = compute_posterior_covariance(Jf_xi)

        # posterior uncertainty in b
        Jf_b = xp.ones(I_data.size) # flattened in q
        std_b = compute_posterior_covariance(Jf_b)

        # posterior uncertainty in w
        std_w_list = []
        for i in range(len(p_dims)):
            w_contract_list = [w for k,w in enumerate(w_list) if k != i]
            Gw_dw = contract_tensor(G, w_contract_list, skip_axes=[*q_axes,p_axes[0]+i])
            Jf_w = xi * Gw_dw
            Jf_w = Jf_w.reshape(-1, Jf_w.shape[-1]) # flatten in q
            std_w_list.append(compute_posterior_covariance(Jf_w))

        # report posterior uncertainty
        print('\n== Parameter Standard Deviation ==')
        print('xi: %.2e' % std_xi)
        print('b: %.2e' % std_b)
        for i,std_w in enumerate(std_w_list):
            print('w'+str(i)+':')
            with xp.printoptions(formatter={'float_kind': '{:.2e}'.format}):
                print(std_w)

        return sens_xi, sens_b, sens_w_list, std_xi, std_b, std_w_list
