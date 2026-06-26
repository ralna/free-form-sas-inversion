"""
Crazy distributions generator

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
from ffsi.array_module import get_array_module

def crazy_distribution(x, gaussians, noise_level, fade_start, fade_end, seed=0):

    # use CPU or GPU as appropriate
    xp = get_array_module(x)

    # create
    w_true = xp.zeros(x.shape)

    # add Gaussians
    for factor, mean, stddev in gaussians:
        w_true += factor * xp.exp(-((x - mean) / stddev) ** 2)

    # add noise
    xp.random.seed(seed)
    w_true += noise_level * xp.random.rand(*x.shape) * xp.random.rand(*x.shape)

    # fade both ends to make it look nicer
    if len(x) >= 3:
        w_true[0:fade_start] = 0.
        w_true[fade_start:fade_end] *= xp.linspace(0, 1, fade_end - fade_start)
        w_true[-fade_start:] = 0.
        w_true[-fade_end:-fade_start] *= xp.linspace(1, 0, fade_end - fade_start)

    # normalize to 1
    w_true /= xp.sum(w_true)
    return w_true
