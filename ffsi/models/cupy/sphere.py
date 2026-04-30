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

    # sphere volume
    V = 4/3 * cp.pi * r ** 3

    # sphere scattering amplitude
    qr = cp.outer(q, r)
    F = 3 * V[None,:] * drho * (cp.sin(qr) - qr * cp.cos(qr)) / qr ** 3

    # Green's function
    return F ** 2
