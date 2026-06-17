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

from ffsi.models.basemodel import SASModel

class Cylinder(SASModel):

    @classmethod
    def compute_G(self, q_list, param_dict, const_dict):

        # extract parameters
        qx, qy = q_list[0], q_list[1]
        l, r = param_dict['l'], param_dict['r']
        theta, phi = param_dict['theta'], param_dict['phi']
        drho = const_dict['drho']

        # use CPU or GPU as appropriate
        xp = cp.get_array_module(qx, qy, l, r, theta, phi, drho)
        xps = cps.get_array_module(qx, qy, l, r, theta, phi, drho)
        print("(using " + xp.__name__ + " and " + xps.__name__ + " for G computation)")

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

        # Green's function (scattering intensity)
        return F ** 2

    @classmethod
    def compute_average_V(self, param_dict, w_dict):

        # extract parameters
        l, r = param_dict['l'], param_dict['r']
        w_l, w_r = w_dict['l'], w_dict['r']

        # use CPU or GPU as appropriate
        xp = cp.get_array_module(l, r, w_l, w_r)

        # cylinder volume
        V = xp.pi * l[:,None] * r[None,:] ** 2

        # average cylinder volume
        return w_l.T @ V @ w_r

    @classmethod
    def get_param_keys_G(self):
        return ['l', 'r', 'theta', 'phi']

    @classmethod
    def get_param_keys_V(self):
        return ['l', 'r']
