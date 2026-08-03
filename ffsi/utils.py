"""
Tensor multiplication utility functions

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
