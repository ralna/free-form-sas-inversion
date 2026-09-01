"""
Get array module that works when CuPy is not installed

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import numpy as _numpy
import scipy as _scipy

from ffsi import CUPY_INSTALLED

def get_array_module(*args):
    if CUPY_INSTALLED:
        import cupy as cp
        return cp.get_array_module(*args)
    else:
        return _numpy

def to_backend(*arrays):
    if CUPY_INSTALLED:
        import cupy as cp

        return tuple(cp.asarray(a) for a in arrays)
    return arrays

def get_science_module(*args):
    if CUPY_INSTALLED:
        import cupyx.scipy as cps
        return cps.get_array_module(*args)
    else:
        return _scipy
