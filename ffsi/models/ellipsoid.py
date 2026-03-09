"""
SAS Ellipsoid Model
https://www.sasview.org/docs/user/models/ellipsoid.html

Parameters:
qx - scattering vector x component
qy - scattering vector y component
rp - polar radius
re - equatorial radius
theta - ellipsoid axis to beam angle
phi - ellipsoid rotation about beam
drho - difference between scattering length densities

Copyright (C) 2025 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import numpy as np

def G_ellipsoid(qx, qy, rp, re, theta, phi, drho):

    # ellipsoid volume
    V = 4/3 * np.pi * rp * re ** 2

    # ellipsoid scattering amplitude
    qc = qx * np.sin(theta) * np.cos(phi) + qy * np.sin(theta) * np.sin(phi)
    qa = np.sqrt(qx ** 2 + qy ** 2 - qc ** 2)
    qr = qa * re + qc * rp
    F = 3 * V * drho * (np.sin(qr) - qr * np.cos(qr)) / qr ** 3

    # Green's function
    return F ** 2
