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

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import cupy as cp

def G_ellipsoid(qx, qy, rp, re, theta, phi, drho):

    # use CPU or GPU as appropriate
    xp = cp.get_array_module(qx, qy, rp, re, theta, phi, drho)
    print("\n (using " + xp.__name__ + " for G computation)")

    # ellipsoid volume
    V = 4/3 * xp.pi * rp[:,None] * re[None,:] ** 2

    # coordinate transformation
    sint_cosp = xp.outer(xp.sin(theta), xp.cos(phi))
    sint_sinp = xp.outer(xp.sin(theta), xp.sin(phi))
    qc = (qx[:,None,None] * sint_cosp[None,:,:])[:,None,:,:] + \
         (qy[:,None,None] * sint_sinp[None,:,:])[None,:,:,:]
    qa = xp.sqrt((qx ** 2)[:,None,None,None] +
                 (qy ** 2)[None,:,None,None] - qc ** 2)

    # ellipsoid scattering amplitude
    qa_re = xp.moveaxis(qa[:,:,:,:,None] * re[None,None,None,None,:], 4, 2)
    qc_rp = xp.moveaxis(qc[:,:,:,:,None] * rp[None,None,None,None,:], 4, 2)
    qr = xp.sqrt((qa_re ** 2)[:,:,None,:,:,:] + (qc_rp ** 2)[:,:,:,None,:,:])

    F = 3 * V[None,None,:,:,None,None] * drho * (xp.sin(qr) - qr * xp.cos(qr)) / qr ** 3

    # Green's function
    return F ** 2
