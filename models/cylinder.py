"""
SAS Cylinder Model
https://www.sasview.org/docs/user/models/cylinder.html

Parameters:
qx - scattering vector x component
qy - scattering vector y component
l - cylinder length
r - cylinder radius
theta - cylinder axis to beam angle
phi - cylinder rotation about beam
drho - difference between scattering length densities

Copyright (C) 2025 The Science and Technology Facilities Council (STFC)
"""
import numpy as np
from scipy.special import j1

def G_cylinder(qx, qy, l, r, theta, phi, drho):

    # cylinder volume
    V = np.pi * l * r ** 2

    # cylinder structure factor
    qc = qx * np.sin(theta) * np.cos(phi) + qy * np.sin(theta) * np.sin(phi)
    qa = np.sqrt(qx ** 2 + qy ** 2 - qc ** 2)
    hqcl = 0.5 * qc * l
    qar = qa * r
    F = 2 * V * drho * ( np.sin(hqcl) / hqcl ) * ( j1(qar) / qar )

    # Green's function
    return F ** 2
