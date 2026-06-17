"""
SAS Sphere Model Test
https://www.sasview.org/docs/user/models/sphere.html

Parameters:
q - scattering vector
r - sphere radius
drho - difference between scattering length densities

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import unittest

import numpy as np
import cupy as cp
from ffsi.models.serial.sphere import G_sphere
from ffsi.models import Sphere


class TestSphere(unittest.TestCase):
    def runTest(self):

        # contrast
        drho = 1

        # q discretisation (log)
        ql = -3
        qu = 0
        nq = 200

        # r discretisation
        rl = 400
        ru = 800
        nr = 500

        # discretise q and r
        q = np.logspace(ql, qu, nq)
        r = np.linspace(rl, ru, nr)

        print('q: logspace(%d,%d,%d)' % (ql, qu, nq))
        print('r: linspace(%d,%d,%d)' % (rl, ru, nr))

        # form Green's function tensor on CPU
        print('\nForming G in serial on CPU...')
        G = np.zeros((nq,nr))
        for iq in range(nq):
            for ir in range(nr):
                G[iq,ir] = G_sphere(q[iq], r[ir], drho)

        # move data to GPU (for testing, normally would be formed on GPU)
        q_gpu = cp.asarray(q)
        r_gpu = cp.asarray(r)

        # form arguments for Green's function computation
        q_list = [q_gpu]
        param_dict = {'r' : r_gpu}
        const_dict = {'drho' : drho}

        # form Green's function tensor on GPU
        print('\nForming G in parallel on GPU...')
        G_gpu = Sphere.compute_G(q_list, param_dict, const_dict)

        # move to CPU for error comparison
        G_cpu = G_gpu.get()

        # compare relative error in G computation
        rel_err = np.linalg.norm(G - G_cpu) / np.linalg.norm(G_cpu)
        print('\nG computation relative error: %.2e' % rel_err)

        self.assertAlmostEqual(rel_err, 0, places=14, msg="Inaccurate Sphere G computation")
