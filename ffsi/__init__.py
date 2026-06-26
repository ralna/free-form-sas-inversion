"""
Free-Form SAS Inversion

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
CUPY_INSTALLED = False

try:
    import cupy as _cupy
    CUPY_INSTALLED = True
    print('INFO: CuPy is installed, GPU computation is available')
except Exception as e:
    print('WARNING: CuPy is not installed, cannot use GPU computation')
    print(e)
    print('WARNING: continuting with CPU computation only')
