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

def get_science_module(*args):
    if CUPY_INSTALLED:
        import cupyx.scipy as cps
        return cps.get_array_module(*args)
    else:
        return _scipy

def to_device(*arrays):
    """
    Bring arrays from the host to the device.
    """
    if CUPY_INSTALLED:
        import cupy as cp
        return (cp.asarray(a) for a in arrays)
    return arrays

def from_device(array):
    """
    Bring an array from the device to the host.
    """
    if CUPY_INSTALLED:
        import cupy as cp
        return cp.asnumpy(array)
    return array
