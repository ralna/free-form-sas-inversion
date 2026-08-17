"""
SAS Cylinder Model (1D)
https://www.sasview.org/docs/user/models/cylinder.html

Parameters:
q - scattering vector
l - cylinder length
r - cylinder radius
drho - difference between scattering length densities

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
from ffsi.array_module import get_array_module, get_science_module
from ffsi.models.utils import gauss_legendre

from ffsi.models.basemodel import SASModel

class Cylinder(SASModel):

    param_names_scattering_intensity = ['l', 'r']

    @staticmethod
    def compute_volume(param_list):
        # extract parameters
        l, r = param_list[0], param_list[1]

        # use CPU or GPU as appropriate
        xp = get_array_module(l, r)

        return xp.pi * l[:, None] * r[None, :] ** 2

    @staticmethod
    def compute_scattering_intensity(q_list, param_list, drho):

        # extract parameters
        q = q_list[0]
        l, r = param_list[0], param_list[1]

        # use CPU or GPU as appropriate
        xp = get_array_module(q, l, r, drho)
        xps = get_science_module(q, l, r, drho)
        print("INFO: using " + xp.__name__ + " and " + xps.__name__ + " for G computation")

        # cylinder volume
        V = Cylinder.compute_volume(param_list)

        # theta is Gauss-Legendre points in [0,pi/2]
        z, w = gauss_legendre(xp) # in [-1,1]
        zm = xp.pi/4 # (b-a)/2
        zb = xp.pi/4 # (a+b)/2
        theta = zm * z + zb # change from [-1,1] to [a,b] integral

        # sine and cosine terms
        sin_theta = xp.sin(theta)
        qs = q[:,None] * sin_theta[None,:]
        qc = q[:,None] * xp.cos(theta)[None,:]

        # cylinder scattering amplitude
        hqcl = 0.5 * xp.moveaxis(qc[:,:,None] * l[None,None,:], 1, 2)
        qsr = xp.moveaxis(qs[:,:,None] * r[None,None,:], 1, 2)
        sin_hqcl = xp.sin(hqcl) / hqcl
        j1_qsr = xps.special.j1(qsr) / qsr
        F = 2 * V[None,:,:,None] * drho * sin_hqcl[:,:,None,:] * j1_qsr[:,None,:,:]

        # integrate scattering intensity over theta using Gaussian quadrature
        F2 = xp.sum(w[None,None,None,:] * (F**2) * sin_theta[None,None,None,:], axis=-1)
        F2 *= zm # change from [-1,1] to [a,b] integral

        # scattering intensity (Green's function)
        return F2

    param_names_average_volume = ['l', 'r']

    @staticmethod
    def compute_average_volume(param_list, w_list):

        # extract parameters
        w_l, w_r = w_list[0], w_list[1]

        # cylinder volume
        V = Cylinder.compute_volume(param_list)

        # average cylinder volume
        return w_l.T @ V @ w_r
