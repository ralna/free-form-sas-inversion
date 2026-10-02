"""
Tests for the ffsi api
"""
import numpy as np
import pytest

from ffsi.api import invert
from ffsi.models.sphere import Sphere

# set ground truth values for simple toy problem, and tolerances for test values.

# discretisation
Q = np.logspace(-3, 0, 320)
R = np.linspace(400.0, 800.0, 200)

SLD = 4.0
SLD_SOLVENT = 3.0
DRHO = SLD - SLD_SOLVENT

SIGMA = {'noiseless': 0.25, 'noisy': 16.0}

def gaussian_weights(grid, centre, width):
    """
    A normalised Gaussian distribution over `grid`.
    """
    weights = np.exp(-(grid - centre) ** 2 / (2 * width ** 2))
    return weights / weights.sum()

# ground truth
SCALE_TRUE = 2.0
BACKGROUND_TRUE = 0.5
MEAN_R_TRUE = 500.0
W_TRUE = gaussian_weights(R, MEAN_R_TRUE, 10.0)
W_PEAK_TRUE = 0.080088220296025706

AVERAGE_VOLUME_TRUE = 5.2422709412901676e8
XI_TRUE = 3.8151404656467128e-13
MEAN_R_VOLUME_TRUE = 500.59952057530961

# simulated intensities, with 20%-30% error bars
G = Sphere.compute_scattering_intensity([Q], [R], DRHO)
I_DATA = XI_TRUE * (G @ W_TRUE) + BACKGROUND_TRUE
I_DATA_STD = (np.random.RandomState(0).rand(Q.size) * 0.1 + 0.2) * I_DATA
I_DATA_NOISY = \
    I_DATA + I_DATA_STD * np.random.default_rng(12345).standard_normal(Q.size)

DATASETS = {'noiseless': I_DATA, 'noisy': I_DATA_NOISY}

CHI2 = {'noiseless': 2.15049897750567e-11, 'noisy': 0.97648549676700314}
RESIDUAL_MAX = {'noiseless': 3.33344810463421e-05, 'noisy': 3.0427813742445928}
RESIDUAL_MIN = {'noiseless': -2.5325843594523642e-05, 'noisy': -2.6138711743103058}

# relative tolerance on the recovered quantities, per dataset
PROBLEM_TOL = [('noiseless', 1e-5), ('noisy', 5e-2)]

@pytest.fixture(scope="module")
def results():
    """
    Fit the toy problem once per dataset, for the whole module.
    """
    return {
        name: invert('sphere', Q, intensity, I_DATA_STD, {'r': R},
                     sld=SLD, sld_solvent=SLD_SOLVENT, sigma=SIGMA[name])
        for name, intensity in DATASETS.items()
    }

class TestInvert:
    """
    Class to test invert() against the sphere toy problem.
    """

    @pytest.mark.parametrize('case', DATASETS)
    def test_invert_returns_correct_drho(self, results, case):
        """
        Test that the contrast used for the fit is reported back
        """
        assert results[case].drho == pytest.approx(1.0)

    @pytest.mark.parametrize('case, rtol', PROBLEM_TOL)
    def test_invert_returns_correct_scale(self, results, case, rtol):
        """
        Test that the volume-fraction scale is recovered
        """
        assert results[case].scale == pytest.approx(SCALE_TRUE, rel=rtol)

    @pytest.mark.parametrize('case, rtol', PROBLEM_TOL)
    def test_invert_returns_correct_xi(self, results, case, rtol):
        """
        Test that xi is recovered

        `abs=0` because xi is ~1e-13, well under the default absolute
        tolerance.
        """
        assert results[case].xi == pytest.approx(XI_TRUE, rel=rtol, abs=0)

    @pytest.mark.parametrize('case, rtol', PROBLEM_TOL)
    def test_invert_returns_correct_background(self, results, case, rtol):
        """
        Test that the flat background is recovered
        """
        assert results[case].background == \
            pytest.approx(BACKGROUND_TRUE, rel=rtol)

    @pytest.mark.parametrize('case, rtol', PROBLEM_TOL)
    def test_invert_returns_correct_average_volume(self, results, case, rtol):
        """
        Test that the average sphere volume is recovered
        """
        assert results[case].average_volume == \
            pytest.approx(AVERAGE_VOLUME_TRUE, rel=rtol)

    @pytest.mark.parametrize('case', DATASETS)
    def test_invert_returns_valid_distribution(self, results, case):
        """
        Test that the weights satisfy the simplex constraint
        """
        weights = results[case].distribution('r').weights

        assert weights.sum() == pytest.approx(1.0, abs=1e-8)
        assert (weights >= 0).all()

    @pytest.mark.parametrize('case', DATASETS)
    def test_invert_returns_correct_chi2(self, results, case):
        """
        Test the mean squared residual against its pinned value
        """
        assert results[case].chi2 == pytest.approx(CHI2[case], abs=1e-6)

    @pytest.mark.parametrize('case', DATASETS)
    def test_invert_returns_correct_residuals(self, results, case):
        """
        Test the residual extremes against their pinned values
        """
        residuals = results[case].residuals

        assert residuals.max() == pytest.approx(RESIDUAL_MAX[case], abs=1e-6)
        assert residuals.min() == pytest.approx(RESIDUAL_MIN[case], abs=1e-6)

    @pytest.mark.parametrize('case', DATASETS)
    def test_invert_returns_correct_result_shapes(self, results, case):
        """
        Test that the fitted curve and residuals are returned on the q grid
        """
        result = results[case]

        assert result.theory.shape == Q.shape
        assert result.residuals.shape == Q.shape
        assert result.distribution('r').grid.shape == R.shape