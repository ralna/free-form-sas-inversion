"""
SAS Ellipsoid Model (1D)
https://www.sasview.org/docs/user/models/ellipsoid.html

Parameters:
q - scattering vector
rp - polar radius
re - equatorial radius
drho - difference between scattering length densities

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
from ffsi.array_module import get_array_module
from ffsi.models.utils import gauss_legendre

from ffsi.models.basemodel import SASModel

class Ellipsoid(SASModel):

    param_names_scattering_intensity = ['rp', 're']

    @staticmethod
    def compute_scattering_intensity(q_list, param_list, drho):

        # extract parameters
        q = q_list[0]
        rp, re = param_list[0], param_list[1]

        # use CPU or GPU as appropriate
        xp = get_array_module(q, rp, re, drho)
        print("INFO: using " + xp.__name__ + " for G computation")

        # ellipsoid volume
        V = 4/3 * xp.pi * rp[:,None] * re[None,:] ** 2

        # u is Gauss-Legendre points in [0,1]
        z, w = gauss_legendre(xp) # in [-1,1]
        zm = 0.5 # (b-a)/2
        zb = 0.5 # (a+b)/2
        u = zm * z + zb # change from [-1,1] to [a,b] integral

        # ellipsoid scattering amplitude
        v2m1 = (rp[:,None] / re[None,:]) ** 2 - 1
        r = re[None,:,None] * xp.sqrt(1 + u[None,None,:]**2 * v2m1[:,:,None])
        qr = q[:,None,None,None] * r[None,:,:,:]

        F = 3 * V[None,:,:,None] * drho * (xp.sin(qr) - qr * xp.cos(qr)) / qr ** 3

        # integrate scattering intensity over u using Gaussian quadrature
        F2 = xp.sum(w[None,None,None,:] * (F**2), axis=-1)
        F2 *= zm # change from [-1,1] to [a,b] integral

        # scattering intensity (Green's function)
        return F2

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
