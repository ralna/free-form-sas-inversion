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

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import numpy as np
from scipy.special import j1

def G_cylinder(qx, qy, l, r, theta, phi, drho):

    # cylinder volume
    V = np.pi * l[:,None] * r[None,:] ** 2

    # coordinate transformation
    sint_cosp = np.outer(np.sin(theta), np.cos(phi))
    sint_sinp = np.outer(np.sin(theta), np.sin(phi))
    qc = (qx[:,None,None] * sint_cosp[None,:,:])[:,None,:,:] + \
         (qy[:,None,None] * sint_sinp[None,:,:])[None,:,:,:]
    qa = np.sqrt((qx ** 2)[:,None,None,None] +
                 (qy ** 2)[None,:,None,None] - qc ** 2)

    # cylinder scattering amplitude
    hqcl = 0.5 * np.moveaxis(qc[:,:,:,:,None] * l[None,None,None,None,:], 4, 2)
    qar = np.moveaxis(qa[:,:,:,:,None] * r[None,None,None,None,:], 4, 2)
    sin_hqcl = np.sin(hqcl) / hqcl
    j1_qar = j1(qar) / qar
    F = 2 * V[None,None,:,:,None,None] * drho * sin_hqcl[:,:,:,None,:,:] * j1_qar[:,:,None,:,:,:]

    # Green's function
    return F ** 2
