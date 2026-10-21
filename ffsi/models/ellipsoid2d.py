"""
SAS Ellipsoid Model (2D)
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
from ffsi.utils import orientation_projections

class Ellipsoid2D(SASModel):

    param_names_scattering_intensity = ['rp', 're', 'theta', 'phi']

    @staticmethod
    def compute_volume(param_list):
        # extract parameters
        rp, re = param_list[0], param_list[1]

        # use CPU or GPU as appropriate
        xp = get_array_module(rp, re)

        # ellipsoid volume
        return 4/3 * xp.pi * rp[:,None] * re[None,:] ** 2

    @staticmethod
    def compute_scattering_intensity(q_list, param_list, drho, paired=False):
        """
        :param q_list: `[qx, qy]`; the axes of a regular grid, or paired detector
            points when `paired` is True
        :param param_list: `[rp, re, theta, phi]`, angles in radians
        :param drho: difference between scattering length densities
        :param paired: evaluate on paired points `(qx[n], qy[n])` instead of the
            `qx` x `qy` grid
        :return: `G` of shape `(nqx, nqy, nrp, nre, ntheta, nphi)`, or
            `(nq, nrp, nre, ntheta, nphi)` when `paired`
        """

        # extract parameters
        qx, qy = q_list[0], q_list[1]
        rp, re = param_list[0], param_list[1]
        theta, phi = param_list[2], param_list[3]

        # use CPU or GPU as appropriate
        xp = get_array_module(qx, qy, rp, re, theta, phi, drho)
        print("INFO: using " + xp.__name__ + " for G computation")

        # ellipsoid volume
        V = Ellipsoid2D.compute_volume(param_list)

        # components of q along (qc) and across (qa) the ellipsoid axis,
        # shape (*q_shape, ntheta, nphi)
        qc, qa = orientation_projections(qx, qy, theta, phi, paired=paired)

        # ellipsoid scattering amplitude, shape (*q_shape, nrp, nre, ntheta, nphi)
        qa_re = qa[..., None, :, :] * re[:, None, None]  # (*q_shape, nre, ntheta, nphi)
        qc_rp = qc[..., None, :, :] * rp[:, None, None]  # (*q_shape, nrp, ntheta, nphi)
        qr = xp.sqrt((qa_re ** 2)[..., None, :, :, :] + (qc_rp ** 2)[..., :, None, :, :])

        # 3 (sin x - x cos x) / x^3, using its series 1 - x^2/10 near 0 where the
        # closed form cancels catastrophically (and is 0/0 at q = 0)
        small = qr < 1e-3
        qr_safe = xp.where(small, 1.0, qr)
        form = xp.where(small, 1 - qr ** 2 / 10,
                        3 * (xp.sin(qr_safe) - qr_safe * xp.cos(qr_safe)) / qr_safe ** 3)
        F = V[:, :, None, None] * drho * form

        # scattering intensity (Green's function)
        return F ** 2

    param_names_average_volume = ['rp', 're']

    @staticmethod
    def compute_average_volume(param_list, w_list):

        # extract parameters
        w_rp, w_re = w_list[0], w_list[1]

        # ellipsoid volume
        V = Ellipsoid2D.compute_volume(param_list)

        # average ellipsoid volume
        return w_rp.T @ V @ w_re
