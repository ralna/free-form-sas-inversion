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

        # integrate over theta in [0,pi/2] using Gauss-Legendre quadrature
        zm = xp.pi/4 # (b-a)/2
        zb = xp.pi/4 # (a+b)/2

        # quadrature for each (q,l,r) depends on q times the largest dimension
        qr_max = q[:,None,None] * xp.maximum(0.5 * l[:,None], r[None,:])[None,:,:]

        # integrate blocks of quadrature points at once
        F2 = xp.zeros(qr_max.shape)
        grid_elements = q.size * max(l.size, r.size) # (q,l) and (q,r) grids below
        for (iq, il, ir), z, w in gauss_legendre(xp, qr_max, elements_per_point=grid_elements):

            # theta is Gauss-Legendre points in [0,pi/2]
            theta = zm * z + zb # change from [-1,1] to [a,b] integral
            sin_theta = xp.sin(theta)
            cos_theta = xp.cos(theta)

            # sine and cosine terms
            qs = sin_theta[:,None] * q[None,:]
            qc = cos_theta[:,None] * q[None,:]

            # the amplitude separates into l and r factors, so evaluate
            # them on the (q,l) and (q,r) grids (j1 is the costly part)
            hqcl = 0.5 * qc[:,:,None] * l[None,None,:]
            qsr = qs[:,:,None] * r[None,None,:]
            sin_hqcl = xp.sin(hqcl) / hqcl
            j1_qsr = xps.special.j1(qsr) / qsr

            # cylinder scattering amplitude (without the volume and contrast)
            F = 2 * sin_hqcl[:, iq, il] * j1_qsr[:, iq, ir]

            # scattering intensity contribution
            F2[iq, il, ir] += (w * sin_theta) @ F**2
        F2 *= zm # change from [-1,1] to [a,b] integral

        # scale by the volume and contrast
        F2 *= (V * drho)[None,:,:] ** 2

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
