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
from ffsi.utils import orientation_projections, safe_sinc

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
    def compute_scattering_intensity(q_list, param_list, drho, paired=False):
        """
        :param q_list: `[qx, qy]`; the axes of a regular grid, or paired detector
            points when `paired` is True
        :param param_list: `[l, r, theta, phi]`, angles in radians
        :param drho: difference between scattering length densities
        :param paired: evaluate on paired points `(qx[n], qy[n])` instead of the
            `qx` x `qy` grid
        :return: `G` of shape `(nqx, nqy, nl, nr, ntheta, nphi)`, or
            `(nq, nl, nr, ntheta, nphi)` when `paired`
        """

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

        # components of q along (qc) and across (qa) the cylinder axis,
        # shape (*q_shape, ntheta, nphi)
        qc, qa = orientation_projections(qx, qy, theta, phi, paired=paired)

        # cylinder scattering amplitude, shape (*q_shape, nl, nr, ntheta, nphi)
        hqcl = 0.5 * qc[..., None, :, :] * l[:, None, None]  # (*q_shape, nl, ntheta, nphi)
        qar = qa[..., None, :, :] * r[:, None, None]  # (*q_shape, nr, ntheta, nphi)
        sin_hqcl = safe_sinc(hqcl)
        # j1(x)/x -> 1/2 as x -> 0 (q along the axis)
        nonzero = qar != 0
        j1_qar = xp.where(nonzero, xps.special.j1(qar) / xp.where(nonzero, qar, 1.0), 0.5)
        F = 2 * V[:, :, None, None] * drho * sin_hqcl[..., :, None, :, :] * j1_qar[..., None, :, :, :]

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
