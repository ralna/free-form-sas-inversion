"""
Low-rank tensor train approximation of Green's function

We use the open source xfac code to perform the tt-approximation:
https://github.com/tensor4all/xfac

Parameters:
G_func - function to compute the Green's tensor
dims - dimensions of each Green's tensor index
tol - tolerance for tt-approximation
max_rank - maximum allowed tt-core rank
compute_true_error - compute true error of approximation (slow)
max_error_evals - maximum evaluations during error computation

Returns:
tt - xfac tensor train

Example usage:

dims = (nq,nr)
G_func = lambda inds: G_sphere(q[inds[0]], r[inds[1]], drho)
tt = tt_approx(G_func, dims)

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import time
# FIXME: compile xfac as part of this module?
# import locally compiled xfac python module
import sys
sys.path.append("../xfac/build/python")
import xfacpy


def tt_approx(G_func, dims, tol, max_rank=250, compute_true_error=False, max_error_evals=1e15):

    print('Computing TT-representation using xfac...')
    print('Tolerance: %.2e' % tol)

    # form low-rank tensor-train approximation
    t0 = time.time()
    param = xfacpy.TensorCI2Param()
    param.reltol = tol
    param.fullPiv = True # more consistent approximation
    param.bondDim = max_rank
    tci = xfacpy.TensorCI2(G_func, dims, param=param)
    while not tci.isDone():
        tci.iterate()
    t1 = time.time()
    print('xfac approximation time: %.2f s' % (t1-t0))

    # print some useful statistics of the representation
    rel_err = tci.pivotError[-1] / tci.pivotError[0]
    print('xfac relative error: %.2e' % rel_err)
    ncores = tci.len()
    print('Number of cores: %d' % ncores)
    print('Core sizes:')
    for i in range(ncores):
        print(tci.tt.core[i].shape)

    # compute true error if requested (slow for large tensors)
    if compute_true_error:

        print('\nComputing TT-representation true error...')
        t0 = time.time()
        abs_err = tci.trueError(max_n_eval=int(max_error_evals))
        t1 = time.time()
        print('xfac true error time: %.2f s' % (t1-t0))

        # compute Frobenius norm(G) for relative error
        import numpy as np
        grid = np.meshgrid(*[np.arange(dim) for dim in dims])
        G_norm = np.sqrt(np.sum(G_func(grid)**2))

        rel_err = abs_err / G_norm
        print('TT-approximation relative error: %.2e' % rel_err)

    return tci.tt
