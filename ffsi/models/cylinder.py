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
import cupy as cp
import cupyx.scipy as cps

def G_cylinder(qx, qy, l, r, theta, phi, drho):

    # use CPU or GPU as appropriate
    xp = cp.get_array_module(qx, qy, l, r, theta, phi, drho)
    xps = cps.get_array_module(qx, qy, l, r, theta, phi, drho)
    print("\n (using " + xp.__name__ + " and " + xps.__name__ + " for G computation)")

    # cylinder volume
    V = xp.pi * l[:,None] * r[None,:] ** 2

    # coordinate transformation
    sint_cosp = xp.outer(xp.sin(theta), xp.cos(phi))
    sint_sinp = xp.outer(xp.sin(theta), xp.sin(phi))
    qc = (qx[:,None,None] * sint_cosp[None,:,:])[:,None,:,:] + \
         (qy[:,None,None] * sint_sinp[None,:,:])[None,:,:,:]
    qa = xp.sqrt((qx ** 2)[:,None,None,None] +
                 (qy ** 2)[None,:,None,None] - qc ** 2)

    # cylinder scattering amplitude
    hqcl = 0.5 * xp.moveaxis(qc[:,:,:,:,None] * l[None,None,None,None,:], 4, 2)
    qar = xp.moveaxis(qa[:,:,:,:,None] * r[None,None,None,None,:], 4, 2)
    sin_hqcl = xp.sin(hqcl) / hqcl
    j1_qar = xps.special.j1(qar) / qar
    F = 2 * V[None,None,:,:,None,None] * drho * sin_hqcl[:,:,:,None,:,:] * j1_qar[:,:,None,:,:,:]

    # Green's function
    return F ** 2
