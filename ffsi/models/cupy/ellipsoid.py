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

    # ellipsoid volume
    V = 4/3 * cp.pi * rp[:,None] * re[None,:] ** 2

    # coordinate transformation
    sint_cosp = cp.outer(cp.sin(theta), cp.cos(phi))
    sint_sinp = cp.outer(cp.sin(theta), cp.sin(phi))
    qc = (qx[:,None,None] * sint_cosp[None,:,:])[:,None,:,:] + \
         (qy[:,None,None] * sint_sinp[None,:,:])[None,:,:,:]
    qa = cp.sqrt((qx ** 2)[:,None,None,None] +
                 (qy ** 2)[None,:,None,None] - qc ** 2)

    # ellipsoid scattering amplitude
    qa_re = cp.moveaxis(qa[:,:,:,:,None] * re[None,None,None,None,:], 4, 2)
    qc_rp = cp.moveaxis(qc[:,:,:,:,None] * rp[None,None,None,None,:], 4, 2)
    qr = cp.sqrt(cp.square(qa_re)[:,:,None,:,:,:] + cp.square(qc_rp)[:,:,:,None,:,:])

    F = 3 * V[None,None,:,:,None,None] * drho * (cp.sin(qr) - qr * cp.cos(qr)) / qr ** 3

    # Green's function
    return F ** 2
