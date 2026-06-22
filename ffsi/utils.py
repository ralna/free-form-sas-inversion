"""
Tensor multiplication utility functions

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import cupy as cp

def contract_tensor(G, w_list, skip_axes):
    """
    Contracts G with a list of 1D weight vectors keeping skipped dimensions.
    """
    if not w_list: # empty weight list
        return G

    # use CPU or GPU as appropriate
    xp = cp.get_array_module(G, w_list)

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

    return xp.einsum(subscripts, G, *w_list)
