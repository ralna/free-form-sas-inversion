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

    @classmethod
    def compute_G(self, q_list, param_dict, const_dict):

        # extract parameters
        q = q_list[0]
        r = param_dict['r']
        drho = const_dict['drho']

        # use CPU or GPU as appropriate
        xp = get_array_module(q, r, drho)
        print("(using " + xp.__name__ + " for G computation)")

        # sphere volume
        V = 4/3 * xp.pi * r ** 3

        # sphere scattering amplitude
        qr = xp.outer(q, r)
        F = 3 * V[None,:] * drho * (xp.sin(qr) - qr * xp.cos(qr)) / qr ** 3

        # Green's function (scattering intensity)
        return F ** 2

    @classmethod
    def compute_average_V(self, param_dict, w_dict):

        # extract parameters
        r = param_dict['r']
        w_r = w_dict['r']

        # use CPU or GPU as appropriate
        xp = get_array_module(r, w_r)

        # sphere volume
        V = 4/3 * xp.pi * r ** 3

        # average sphere volume
        return V @ w_r

    @classmethod
    def get_param_keys_G(self):
        return ['r']

    @classmethod
    def get_param_keys_V(self):
        return ['r']
