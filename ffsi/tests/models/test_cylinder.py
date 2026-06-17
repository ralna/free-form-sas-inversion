"""
SAS Cylinder Model Test
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
import unittest

import numpy as np
import cupy as cp
from ffsi.models.serial.cylinder import G_cylinder as G_cylinder
from ffsi.models import Cylinder


class TestCylinder(unittest.TestCase):
    def runTest(self):

        # contrast
        drho = 1

        # qx, qy discretisation
        nqx = 30
        nqy = 30
        q_side = np.logspace(-2, 0, 15) # log scale on each side
        qx = np.hstack((-q_side[::-1], q_side))
        qy = qx.copy()

        # l discretisation
        ll = 200
        lu = 600
        nl = 10

        # r discretisation
        rl = 50
        ru = 90
        nr = 9

        # theta discretisation
        thetal = 20
        thetau = 75
        ntheta = 8

        # phi discretisation
        phil = 150
        phiu = 240
        nphi = 7

        # discretise l, r, theta, phi
        l = np.linspace(ll, lu, nl)
        r = np.linspace(rl, ru, nr)
        theta = np.linspace(thetal, thetau, ntheta)
        phi = np.linspace(phil, phiu, nphi)

        print('qx: %d' % nqx)
        print('qy: %d' % nqy)
        print('l: linspace(%d,%d,%d)' % (ll, lu, nl))
        print('r: linspace(%d,%d,%d)' % (rl, ru, nr))
        print('theta: linspace(%d,%d,%d)' % (thetal, thetau, ntheta))
        print('phi: linspace(%d,%d,%d)' % (phil, phiu, nphi))

        # form Green's function tensor on CPU
        print('\nForming G in serial on CPU...')
        G = np.zeros((nqx,nqy,nl,nr,ntheta,nphi))
        for iqx in range(nqx):
            print('  progress at iqx %d out of %d' % (iqx+1,nqx))
            for iqy in range(nqy):
                for il in range(nl):
                    for ir in range(nr):
                        for it in range(ntheta):
                            for ip in range(nphi):
                                G[iqx,iqy,il,ir,it,ip] = G_cylinder(qx[iqx], qy[iqy], l[il], r[ir], theta[it], phi[ip], drho)

        # move data to GPU (for testing, normally would be formed on GPU)
        qx_gpu = cp.asarray(qx)
        qy_gpu = cp.asarray(qy)
        l_gpu = cp.asarray(l)
        r_gpu = cp.asarray(r)
        theta_gpu = cp.asarray(theta)
        phi_gpu = cp.asarray(phi)

        # form arguments for Green's function computation
        q_list = [qx_gpu, qy_gpu]
        param_dict = {'l': l_gpu, 'r' : r_gpu, 'theta' : theta_gpu, 'phi' : phi_gpu}
        const_dict = {'drho' : drho}

        # form Green's function tensor on GPU
        print('\nForming G in parallel on GPU...')
        G_gpu = Cylinder.compute_G(q_list, param_dict, const_dict)

        # move to CPU for error comparison
        G_cpu = G_gpu.get()

        # compare relative error in G computation
        rel_err = np.linalg.norm(G - G_cpu) / np.linalg.norm(G_cpu)
        print('\nG computation relative error: %.2e' % rel_err)

        self.assertAlmostEqual(rel_err, 0, places=14, msg="Inaccurate Cylinder G computation")
