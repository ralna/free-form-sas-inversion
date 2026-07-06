"""
SAS Sphere Model
https://www.sasview.org/docs/user/models/sphere.html

Parameters:
q - scattering vector
r - sphere radius
drho - difference between scattering length densities

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
from ffsi.array_module import get_array_module

from ffsi.models.basemodel import SASModel

class Sphere(SASModel):

    param_names_scattering_intensity = ['r']

    @staticmethod
    def compute_scattering_intensity(q_list, param_list, drho):

        # extract parameters
        q = q_list[0]
        r = param_list[0]

        # use CPU or GPU as appropriate
        xp = get_array_module(q, r, drho)
        print("INFO: using " + xp.__name__ + " for G computation")

        # sphere volume
        V = 4/3 * xp.pi * r ** 3

        # sphere scattering amplitude
        qr = xp.outer(q, r)
        F = 3 * V[None,:] * drho * (xp.sin(qr) - qr * xp.cos(qr)) / qr ** 3

        # scattering intensity (Green's function)
        return F ** 2

    param_names_average_volume = ['r']

    @staticmethod
    def compute_average_volume(param_list, w_list):

        # extract parameters
        r = param_list[0]
        w_r = w_list[0]

        # use CPU or GPU as appropriate
        xp = get_array_module(r, w_r)

        # sphere volume
        V = 4/3 * xp.pi * r ** 3

        # average sphere volume
        return V @ w_r
