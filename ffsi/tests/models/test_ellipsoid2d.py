"""
SAS Ellipsoid Model Test
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
import unittest

import numpy as np
import cupy as cp
from ffsi.models.serial.ellipsoid2d import G_ellipsoid2d
from ffsi.models import Ellipsoid2D


class TestEllipsoid(unittest.TestCase):
    def runTest(self):

        # contrast
        drho = 1

        # qx, qy discretisation
        nqx = 30
        nqy = 30
        q_side = np.logspace(-2, 0, 15) # log scale on each side
        qx = np.hstack((-q_side[::-1], q_side))
        qy = qx.copy()

        # rp discretisation
        rpl = 50
        rpu = 90
        nrp = 10

        # re discretisation
        rel = 200
        reu = 600
        nre = 9

        # theta discretisation
        thetal = 20
        thetau = 75
        ntheta = 8

        # phi discretisation
        phil = 150
        phiu = 240
        nphi = 7

        # discretise l, r, theta, phi
        rp = np.linspace(rpl, rpu, nrp)
        re = np.linspace(rel, reu, nre)
        theta = np.linspace(thetal, thetau, ntheta)
        phi = np.linspace(phil, phiu, nphi)

        print('qx: %d' % nqx)
        print('qy: %d' % nqy)
        print('rp: linspace(%d,%d,%d)' % (rpl, rpu, nrp))
        print('re: linspace(%d,%d,%d)' % (rel, reu, nre))
        print('theta: linspace(%d,%d,%d)' % (thetal, thetau, ntheta))
        print('phi: linspace(%d,%d,%d)' % (phil, phiu, nphi))

        # form Green's function tensor on CPU
        print('\nForming G in serial on CPU...')
        G = np.zeros((nqx,nqy,nrp,nre,ntheta,nphi))
        for iqx in range(nqx):
            print('  progress at iqx %d out of %d' % (iqx+1,nqx))
            for iqy in range(nqy):
                for irp in range(nrp):
                    for ire in range(nre):
                        for it in range(ntheta):
                            for ip in range(nphi):
                                G[iqx,iqy,irp,ire,it,ip] = G_ellipsoid2d(qx[iqx], qy[iqy], rp[irp], re[ire], theta[it], phi[ip], drho)

        # move data to GPU (for testing, normally would be formed on GPU)
        qx_gpu = cp.asarray(qx)
        qy_gpu = cp.asarray(qy)
        rp_gpu = cp.asarray(rp)
        re_gpu = cp.asarray(re)
        theta_gpu = cp.asarray(theta)
        phi_gpu = cp.asarray(phi)

        # form arguments for Green's function computation
        q_list = [qx_gpu, qy_gpu]
        param_list = [rp_gpu, re_gpu, theta_gpu, phi_gpu]

        # form Green's function tensor on GPU
        print('\nForming G in parallel on GPU...')
        G_gpu = Ellipsoid2D.compute_scattering_intensity(q_list, param_list, drho)

        # move to CPU for error comparison
        G_cpu = G_gpu.get()

        # compare relative error in G computation
        rel_err = np.linalg.norm(G - G_cpu) / np.linalg.norm(G_cpu)
        print('\nG computation relative error: %.2e' % rel_err)

        self.assertAlmostEqual(rel_err, 0, places=14, msg="Inaccurate Ellipsoid G computation")
