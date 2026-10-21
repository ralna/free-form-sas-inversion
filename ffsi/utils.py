"""
Tensor multiplication and other utility functions

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
from ffsi.array_module import get_array_module

def contract_tensor(G, w_list, skip_axes):
    """
    Contracts G with a list of 1D weight vectors keeping skipped dimensions
    """
    if not w_list: # empty weight list
        return G

    # use CPU or GPU as appropriate
    xp = get_array_module(G, w_list)

    # create label string for G (e.g. 'abcd')
    g_labels = [chr(97+i) for i in range(G.ndim)]
    g_str = "".join(g_labels)

    # identify contracted dimensions
    w_labels = [l for i,l in enumerate(g_labels) if i not in skip_axes]
    w_str = ",".join(w_labels)

    # identify output dimensions
    out_labels = [l for i,l in enumerate(g_labels) if i in skip_axes]
    out_str = "".join(out_labels)

    # construct einsum formula (e.g. 'abcd,cd->ab')
    subscripts = f"{g_str},{w_str}->{out_str}"

    return xp.einsum(subscripts, G, *w_list, optimize=True)

def smear_tensor_1d(G, w):
    """
    Smears G with a matrix of weights (1d q case)
    """

    # use CPU or GPU as appropriate
    xp = get_array_module(G, w)

    # smear tensor with weight matrix
    return xp.einsum('i..., ij -> j...', G, w, optimize=True)

def smear_tensor_2d(G, w):
    """
    Smears G with a vector of Gaussian weights (2d q case)
    """

    # use CPU or GPU as appropriate
    xp = get_array_module(G, w)

    # reshape (nqx_calc, nqy_calc) to (nbins, nqx, nbins, nqy)
    nbins = w.shape[0]
    nqx = int(G.shape[0] / nbins)
    nqy = int(G.shape[1] / nbins)
    G_reshaped = xp.reshape(G, (nbins, nqx, nbins, nqy, *G.shape[2:]))

    # smear tensor in x and y with Gaussian weight vector and normalize
    return xp.einsum('ijkl...,i,k -> jl...', G_reshaped, w, w, optimize=True) / (xp.sum(w) ** 2)

def orientation_projections(qx, qy, theta, phi, paired=False):
    """
    Project 2D scattering vectors onto the particle axis for every orientation.

    The particle axis is `(sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))`
    and q lies in the detector plane `(qx, qy, 0)` (the sasmodels convention).

    :param qx: x components of q (1D)
    :param qy: y components of q (1D)
    :param theta: axis-to-beam angles in radians (1D)
    :param phi: axis rotations about the beam in radians (1D)
    :param paired: if True, `qx[n], qy[n]` are paired detector pixels (equal
        lengths) and the q shape is `(nq,)`; otherwise `qx` and `qy` are the axes
        of a regular grid and the q shape is `(nqx, nqy)`
    :return: `(qc, qa)`, the components of q along and perpendicular to the
        axis, each of shape `(*q_shape, ntheta, nphi)`
    """

    # use CPU or GPU as appropriate
    xp = get_array_module(qx, qy, theta, phi)

    sint_cosp = xp.outer(xp.sin(theta), xp.cos(phi))
    sint_sinp = xp.outer(xp.sin(theta), xp.sin(phi))

    if paired:
        qx_b, qy_b = qx[:, None, None], qy[:, None, None]
    else:
        qx_b, qy_b = qx[:, None, None, None], qy[None, :, None, None]

    qc = qx_b * sint_cosp + qy_b * sint_sinp
    # clamp: rounding makes q**2 - qc**2 slightly negative when q is along the axis
    qa = xp.sqrt(xp.maximum(qx_b ** 2 + qy_b ** 2 - qc ** 2, 0.0))
    return qc, qa

def safe_sinc(x):
    """
    `sin(x)/x` with the limit 1 at x = 0
    """
    xp = get_array_module(x)
    return xp.sinc(x / xp.pi)

def xi_to_scale(xi, average_volume):
    """
    Convert `xi` to SasView volume-fraction `scale`.
    """
    return (xi * average_volume) / 1e-4

def scale_to_xi(scale, average_volume):
    """
    Convert SasView volume-fraction `scale` to `xi`.
    """
    return 1e-4 * scale / average_volume
