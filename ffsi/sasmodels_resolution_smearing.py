import numpy as np

from sasmodels.resolution import Pinhole1D, Slit1D
from sasmodels.resolution2d import Pinhole2D
from sasmodels.data import Data2D

from sasmodels.resolution import TEST_PARS_PINHOLE_SPHERE, TEST_DATA_PINHOLE_SPHERE
from sasmodels.resolution import TEST_PARS_SLIT_SPHERE, TEST_DATA_SLIT_SPHERE


class Test1DSphere:
    """
    Test resolution calculations against those returned by Igor.
    Note: these are the tests in SASModels, they are really bad!
    It would be better to test against SASModels rather than Igor.
    """

    def __init__(self):
        """
        Initialise 1D free-form sphere model
        """
        from ffsi.models import Sphere
        self.model = Sphere()

    def _eval_sphere_smeared(self, pars, resolution):
        """
        Evaluate smeared forward model for the sphere
        pars - sasmodels sphere parameters dictionary
        resolution - sasmodels 1D resolution instance
        """
        # extract model parameters (need dummy second parameter as code is vectorized)
        param_list = [np.array([pars['radius'],50])] # dummy second parameter of 50
        drho = pars['sld_solvent'] - pars['sld']

        # convert scale to xi
        w_list = [np.array([1.0,0.0])] # dummy second parameter of 0
        V_ave = self.model.compute_average_volume(param_list, w_list)
        xi = 1e-4 * pars['scale'] / V_ave

        # compute smeared G
        G = self.model.compute_smeared_scattering_intensity([resolution.q_calc], resolution.weight_matrix, param_list, drho)
        print('G smeared shape', G.shape)

        # compute smeared forward model for the sphere
        result = xi * G + pars['background']
        return result[:,0] # for the test we compare to the first parameter

    def _compare(self, output, answer, tolerance):
        np.testing.assert_allclose(output, answer, rtol=tolerance)

    # Pinhole smearing (1D)
    def test_pinhole1d(self):
        """
        Compare 1D pinhole resolution smearing with NIST Igor SANS
        """
        print('=== 1D Pinhole Test ===')

        pars = TEST_PARS_PINHOLE_SPHERE
        data_string = TEST_DATA_PINHOLE_SPHERE

        # load pinhole smeared Igor SANS data
        data = np.loadtxt(data_string.split('\n')).T
        q, q_width, I_data = data
        print('q shape:', q.shape)

        # compute 1d pinhole smearing
        resolution = Pinhole1D(q, q_width)
        I_data_smeared = self._eval_sphere_smeared(pars, resolution)

        print('q_calc shape:', resolution.q_calc.shape)
        print('w shape:', resolution.weight_matrix.shape)
        print('I_data_smeared shape:', I_data_smeared.shape)

        # compare to NIST Igor data
        self._compare(I_data_smeared, I_data, 1e-2)

    # Slit smearing (1D)
    def test_slit1d(self):
        """
        Compare 1D slit resolution smearing with NIST Igor SANS
        """
        print('=== 1D Slit Test ===')

        pars = TEST_PARS_SLIT_SPHERE
        data_string = TEST_DATA_SLIT_SPHERE

        # load slit smeared Igor SANS data
        data = np.loadtxt(data_string.split('\n')).T
        q, delta_qv, _, I_data = data
        print('q shape:', q.shape)

        # compute 1d slit smearing
        resolution = Slit1D(q, q_length=delta_qv, q_width=0)
        I_data_smeared = self._eval_sphere_smeared(pars, resolution)

        print('q_calc shape:', resolution.q_calc.shape)
        print('w shape:', resolution.weight_matrix.shape)
        print('I_data_smeared shape:', I_data_smeared.shape)

        # compare to NIST Igor data
        self._compare(I_data_smeared, I_data, 0.5)


class Test2DCylinder:

    # Pinhole smearing (2D)
    def test_pinhole2d(self):

        print('=== 2D Pinhole Test ===')

        # qx, qy discretisation
        q_side = np.logspace(-2, 0, 15) # log scale on each side
        qx = np.hstack((-q_side[::-1], q_side))
        qy = qx.copy()
        q_list = [qx,qy]

        print('q shape:', qx.shape, qy.shape)

        # simulate some pinhole smear dqx, dqy (no idea if these make sense)
        dqx = 0.0016060 * np.ones(len(qx))
        dqy = dqx.copy()

        # simulate some parameter data
        l = np.array([400,500])
        r = np.array([20,50])
        theta = np.array([np.pi/3, np.pi/2])
        phi = np.array([np.pi/3, np.pi/2])
        param_list = [l, r, theta, phi]
        drho = 3

        # toy parameter distributions
        w_l_true = np.array([0.5,0.5])
        w_r_true = np.array([0.5,0.5])
        w_theta_true = np.array([0.5,0.5])
        w_phi_true = np.array([0.5,0.5])

        # ground truth of scale and background
        scale_true = 0.15
        b_true = 2.2e-4

        # instantiate 2D cylinder model
        from ffsi.models import Cylinder2D
        sasmodel = Cylinder2D()

        # compute the ground truth of xi
        vol_param_list = [l, r]
        vol_w_true_list = [w_l_true, w_r_true]
        V_ave = sasmodel.compute_average_volume(vol_param_list, vol_w_true_list)
        xi_true = 1e-4 * scale_true / V_ave

        # compute true G
        G = sasmodel.compute_scattering_intensity(q_list, param_list, drho)
        print('G shape', G.shape)

        # compute forward model for the 2d cylinder
        from ffsi.utils import contract_tensor
        w_true_list = [w_l_true, w_r_true, w_theta_true, w_phi_true]
        Gw_true = contract_tensor(G, w_true_list, skip_axes=[0,1])
        I_data = xi_true * Gw_true + b_true # simulated I_data

        # wrap q data in sasmodels 2D data class for Pinhole2D class
        q_data = Data2D(x=qx, y=qy, dx=dqx, dy=dqy, z=I_data)

        # do 2d pinhole smearing
        resolution = Pinhole2D(data=q_data, index=None, nsigma=3.0)

        print('q_calc shape:', resolution.q_calc[0].shape, resolution.q_calc[1].shape)
        print('w shape:', resolution.q_calc_weights.shape)

        # compute smeared G
        G = sasmodel.compute_smeared_scattering_intensity(resolution.q_calc, resolution.q_calc_weights, param_list, drho)
        print('G smeared shape', G.shape)

        # compute smeared I_data
        I_data_smeared = xi_true * G + b_true
        print('I_data_smeared shape:', I_data_smeared.shape)

# run 1D tests
test = Test1DSphere()
test.test_pinhole1d()
test.test_slit1d()

# run 2D tests
test = Test2DCylinder()
test.test_pinhole2d()
