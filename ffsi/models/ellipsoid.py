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
from ffsi.array_module import get_array_module

from ffsi.models.basemodel import SASModel

class Ellipsoid(SASModel):

    param_names_scattering_intensity = ['rp', 're', 'theta', 'phi']

    @staticmethod
    def compute_scattering_intensity(q_list, param_list, drho):

        # extract parameters
        qx, qy = q_list[0], q_list[1]
        rp, re = param_list[0], param_list[1]
        theta, phi = param_list[2], param_list[3]

        # use CPU or GPU as appropriate
        xp = get_array_module(qx, qy, rp, re, theta, phi, drho)
        print("INFO: using " + xp.__name__ + " for G computation")

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

        # scattering intensity (Green's function)
        return F ** 2

    param_names_average_volume = ['rp', 're']

    @staticmethod
    def compute_average_volume(param_list, w_list):

        # extract parameters
        rp, re = param_list[0], param_list[1]
        w_rp, w_re = w_list[0], w_list[1]

        # use CPU or GPU as appropriate
        xp = get_array_module(rp, re, w_rp, w_re)

        # ellipsoid volume
        V = 4/3 * xp.pi * rp[:,None] * re[None,:] ** 2

        # average ellipsoid volume
        return w_rp.T @ V @ w_re
