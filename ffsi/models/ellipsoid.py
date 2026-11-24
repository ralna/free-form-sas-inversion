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
    def compute_volume(param_list):

        # extract parameters
        rp, re = param_list[0], param_list[1]

        # use CPU or GPU as appropriate
        xp = get_array_module(rp, re)

        # ellipsoid volume
        return 4/3 * xp.pi * rp[:,None] * re[None,:] ** 2

    @staticmethod
    def compute_scattering_intensity(q_list, param_list, drho):

        # extract parameters
        q = q_list[0]
        rp, re = param_list[0], param_list[1]

        # use CPU or GPU as appropriate
        xp = get_array_module(q, rp, re, drho)
        print("INFO: using " + xp.__name__ + " for G computation")

        # ellipsoid volume
        V = Ellipsoid.compute_volume(param_list)

        # integrate over u in [0,1] using Gauss-Legendre quadrature
        zm = 0.5 # (b-a)/2
        zb = 0.5 # (a+b)/2

        # quadrature for each (q,rp,re) depends on q times the largest radius
        qr_max = q[:,None,None] * xp.maximum(rp[:,None], re[None,:])[None,:,:]

        # integrate blocks of quadrature points at once
        F2 = xp.zeros(qr_max.shape)
        for (iq, irp, ire), z, w in gauss_legendre(xp, qr_max):

            # u is Gauss-Legendre points in [0,1]
            u = zm * z + zb # change from [-1,1] to [a,b] integral

            # ellipsoid scattering amplitude (without the volume and contrast)
            v2m1 = (rp[irp] / re[ire]) ** 2 - 1
            r = re[ire][None,:] * xp.sqrt(1 + u[:,None]**2 * v2m1[None,:])
            qr = q[iq][None,:] * r
            F = 3 * (xp.sin(qr) - qr * xp.cos(qr)) / qr ** 3

            # scattering intensity contribution
            F2[iq, irp, ire] += w @ F**2
        F2 *= zm # change from [-1,1] to [a,b] integral

        # scale by the volume and contrast
        F2 *= (V * drho)[None,:,:] ** 2

        # scattering intensity (Green's function)
        return F2

    param_names_average_volume = ['rp', 're']

    @staticmethod
    def compute_average_volume(param_list, w_list):

        # extract parameters
        w_rp, w_re = w_list[0], w_list[1]

        # ellipsoid volume
        V = Ellipsoid.compute_volume(param_list)

        # average ellipsoid volume
        return w_rp.T @ V @ w_re
