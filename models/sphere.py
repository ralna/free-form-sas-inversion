"""
SAS Sphere Model
https://www.sasview.org/docs/user/models/sphere.html

Copyright (C) 2025 The Science and Technology Facilities Council (STFC)
"""
import numpy as np

def G(q, r, drho):

    # sphere volume
    V = 4/3 * np.pi * r ** 3

    # sphere structure factor
    qr = q * r
    F = 3 * V * drho * (np.sin(qr) - qr * np.cos(qr)) / qr ** 3

    # Green's function
    return F ** 2
