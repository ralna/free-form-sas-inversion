"""
SAS Cylinder Model (2D)
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
from ffsi.array_module import get_array_module, get_science_module

from ffsi.models.basemodel import SASModel

class Cylinder2D(SASModel):

    param_names_scattering_intensity = ['l', 'r', 'theta', 'phi']

    @staticmethod
    def compute_volume(param_list):
        # extract parameters
        l, r = param_list[0], param_list[1]

        # use CPU or GPU as appropriate
        xp = get_array_module(l, r)

        # cylinder volume
        return xp.pi * l[:, None] * r[None, :] ** 2

    @staticmethod
    def compute_scattering_intensity(q_list, param_list, drho):

        # extract parameters
        qx, qy = q_list[0], q_list[1]
        l, r = param_list[0], param_list[1]
        theta, phi = param_list[2], param_list[3]

        # use CPU or GPU as appropriate
        xp = get_array_module(qx, qy, l, r, theta, phi, drho)
        xps = get_science_module(qx, qy, l, r, theta, phi, drho)
        print("INFO: using " + xp.__name__ + " and " + xps.__name__ + " for G computation")

        # cylinder volume
        V = Cylinder2D.compute_volume(param_list)

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

        # scattering intensity (Green's function)
        return F ** 2

    param_names_average_volume = ['l', 'r']

    @staticmethod
    def compute_average_volume(param_list, w_list):

        # extract parameters
        w_l, w_r = w_list[0], w_list[1]

        # cylinder volume
        V = Cylinder2D.compute_volume(param_list)

        # average cylinder volume
        return w_l.T @ V @ w_r
