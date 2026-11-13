"""
Tests for the 2D route of `ffsi.api.invert` (cylinder2d, ellipsoid2d): mocked
plumbing tests, forward-model and resolution-smearing checks, plus solver tests
against a synthetic 2D cylinder problem with known ground truth.
"""
import numpy as np
import pytest

from ffsi.api import invert
from ffsi.models.cylinder2d import Cylinder2D
from ffsi.utils import contract_tensor, scale_to_xi

# -------------------------------- 2D cylinder on synthetic data (pinned numerics)

# discretisation (even lengths keep q = 0 off the grid)
Q2D = np.linspace(-0.12, 0.12, 24)
L = np.linspace(100.0, 140.0, 3)
R = np.linspace(20.0, 30.0, 3)
THETA = np.deg2rad(np.linspace(20.0, 80.0, 4))
PHI = np.deg2rad(np.linspace(0.0, 180.0, 4))
GRIDS_2D = {"l": L, "r": R, "theta": THETA, "phi": PHI}

SLD = 4.0
SLD_SOLVENT = 1.0
SIGMA_2D = 0.25

# ground truth: every particle has L = 120, r = 25, theta = 40 deg, phi = 60 deg
W_TRUE_2D = [np.eye(3)[1], np.eye(3)[1], np.eye(4)[1], np.eye(4)[1]]
TRUE_INDEX = [1, 1, 1, 1]
SCALE_TRUE_2D = 0.01
BACKGROUND_TRUE_2D = 0.001

AVERAGE_VOLUME_TRUE_2D = np.pi * 120.0 * 25.0**2
XI_TRUE_2D = scale_to_xi(SCALE_TRUE_2D, AVERAGE_VOLUME_TRUE_2D)

# simulated images, with 1% error bars
_G2D = Cylinder2D.compute_scattering_intensity([Q2D, Q2D], [L, R, THETA, PHI], SLD - SLD_SOLVENT)
I_2D = XI_TRUE_2D * contract_tensor(_G2D, W_TRUE_2D, skip_axes=[0, 1]) + BACKGROUND_TRUE_2D
I_2D_STD = 0.01 * I_2D
I_2D_NOISY = I_2D + I_2D_STD * np.random.default_rng(12345).standard_normal(I_2D.shape)

DATASETS_2D = {"noiseless": I_2D, "noisy": I_2D_NOISY}

CHI2_2D = {"noiseless": 6.57031e-12, "noisy": 1.005748}
RESIDUAL_MAX_2D = {"noiseless": 1.68536e-05, "noisy": 3.105851}
RESIDUAL_MIN_2D = {"noiseless": -9.92617e-06, "noisy": -3.242743}

PROBLEM_TOL_2D = [("noiseless", 1e-6), ("noisy", 1e-2)]


@pytest.fixture(scope="module")
def results_2d():
    """
    Fit the 2D toy problem once per dataset, for the whole module.
    """
    return {
        name: invert("cylinder2d", (Q2D, Q2D), intensity, I_2D_STD, GRIDS_2D,
                     sld=SLD, sld_solvent=SLD_SOLVENT, sigma=SIGMA_2D)
        for name, intensity in DATASETS_2D.items()
    }


class TestInvert2D:
    """
    Run invert() end to end (real GALAHAD solver, no mocking) on a synthetic
    oriented-cylinder image with known ground truth, for noiseless and noisy
    data, and check that the recovered scale, xi, background, average volume,
    sizes and orientation match the truth, and that chi2 and residuals match
    their pinned values.
    """

    @pytest.mark.parametrize("case", DATASETS_2D)
    def test_invert_returns_correct_drho(self, results_2d, case):
        """
        Test that the contrast used for the fit is reported back
        """
        assert results_2d[case].drho == pytest.approx(3.0)

    @pytest.mark.parametrize("case, rtol", PROBLEM_TOL_2D)
    def test_invert_returns_correct_scale(self, results_2d, case, rtol):
        """
        Test that the volume-fraction scale is recovered
        """
        assert results_2d[case].scale == pytest.approx(SCALE_TRUE_2D, rel=rtol)

    @pytest.mark.parametrize("case, rtol", PROBLEM_TOL_2D)
    def test_invert_returns_correct_xi(self, results_2d, case, rtol):
        """
        Test that xi is recovered (`abs=0`: xi is ~1e-12)
        """
        assert results_2d[case].xi == pytest.approx(XI_TRUE_2D, rel=rtol, abs=0)

    @pytest.mark.parametrize("case, rtol", PROBLEM_TOL_2D)
    def test_invert_returns_correct_background(self, results_2d, case, rtol):
        """
        Test that the flat background is recovered
        """
        assert results_2d[case].background == pytest.approx(BACKGROUND_TRUE_2D, rel=rtol)

    @pytest.mark.parametrize("case, rtol", PROBLEM_TOL_2D)
    def test_invert_returns_correct_average_volume(self, results_2d, case, rtol):
        """
        Test that the average cylinder volume is recovered
        """
        assert results_2d[case].average_volume == pytest.approx(AVERAGE_VOLUME_TRUE_2D, rel=rtol)

    @pytest.mark.parametrize("case", DATASETS_2D)
    def test_invert_returns_valid_distributions(self, results_2d, case):
        """
        Test that every marginal satisfies the simplex constraint
        """
        for dist in results_2d[case].distributions:
            assert dist.weights.sum() == pytest.approx(1.0, abs=1e-8)
            assert (dist.weights >= -1e-10).all()

    @pytest.mark.parametrize("case", DATASETS_2D)
    def test_invert_recovers_size_and_orientation(self, results_2d, case):
        """
        Test that the generating length, radius, theta and phi carry the
        largest weight in their marginals
        """
        result = results_2d[case]

        assert [int(np.argmax(d.weights)) for d in result.distributions] == TRUE_INDEX

    @pytest.mark.parametrize("case", DATASETS_2D)
    def test_invert_returns_correct_chi2(self, results_2d, case):
        """
        Test the mean squared residual against its pinned value
        """
        assert results_2d[case].chi2 == pytest.approx(CHI2_2D[case], abs=1e-6)

    @pytest.mark.parametrize("case", DATASETS_2D)
    def test_invert_returns_correct_residuals(self, results_2d, case):
        """
        Test the residual extremes against their pinned values
        """
        residuals = results_2d[case].residuals

        assert residuals.max() == pytest.approx(RESIDUAL_MAX_2D[case], abs=1e-6)
        assert residuals.min() == pytest.approx(RESIDUAL_MIN_2D[case], abs=1e-6)

    @pytest.mark.parametrize("case", DATASETS_2D)
    def test_invert_returns_correct_result_shapes(self, results_2d, case):
        """
        Test that the fitted image and residuals are on the (qx, qy) grid
        """
        result = results_2d[case]

        assert result.theory.shape == (len(Q2D), len(Q2D))
        assert result.residuals.shape == (len(Q2D), len(Q2D))