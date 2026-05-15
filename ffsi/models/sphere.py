"""
SAS Sphere Model
https://www.sasview.org/docs/user/models/sphere.html

Parameters:
q - scattering vector
r - sphere radius
drho - difference between scattering length densities

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import cupy as cp

def G_sphere(q, r, drho):

    # use CPU or GPU as appropriate
    xp = cp.get_array_module(q, r, drho)
    print("\n (using " + xp.__name__ + " for G computation)")

    # sphere volume
    V = 4/3 * xp.pi * r ** 3

    # sphere scattering amplitude
    qr = xp.outer(q, r)
    F = 3 * V[None,:] * drho * (xp.sin(qr) - qr * xp.cos(qr)) / qr ** 3

    # Green's function
    return F ** 2
