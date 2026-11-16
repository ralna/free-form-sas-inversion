"""
Mock-based plumbing tests for `ffsi.api.invert`, plus solver tests against a
synthetic toy problem with known ground truth.
"""
import numpy as np
import pytest

from ffsi.models.cylinder2d import Cylinder2D
from ffsi.utils import contract_tensor, scale_to_xi
from ffsi.api import ParamDistribution, InversionResult, _resolve_model, invert
from ffsi.models.sphere import Sphere


@pytest.fixture(autouse=True)
def force_cpu(monkeypatch):
    """Run on numpy so G is a host array and results are deterministic."""
    monkeypatch.setattr("ffsi.array_module.CUPY_INSTALLED", False)


class RecordingOptimize:
    """
    Stand-in for `ffsi.optimize_galahad.optimize`.

    Returns a deterministic simplex distribution per model parameter, sized from
    G (axis 0 is q, the remaining axes are the parameter grids), and records the
    G shape and sigma it was called with so tests can assert the solver contract.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, G, I_data, I_data_std, sigma=None):
        self.calls.append({"G_shape": tuple(G.shape), "sigma": sigma})
        w_list = [np.full(G.shape[ax], 1.0 / G.shape[ax]) for ax in range(1, G.ndim)]
        return 1.5, 0.1, w_list


@pytest.fixture
def fake_optimize(monkeypatch):
    """Patch the solver reference bound into ffsi.api; return the recorder."""
    recorder = RecordingOptimize()
    monkeypatch.setattr("ffsi.api.optimize", recorder)
    return recorder


def _sphere_data(n=32):
    """A plain descending curve; the mock ignores the values, so any will do."""
    q = np.linspace(0.005, 0.2, n)
    iq = np.linspace(10.0, 1.0, n)
    diq = 0.1 * iq
    return q, iq, diq


def _invert_sphere(grid=(10.0, 100.0, 16), **kwargs):
    """`invert` on `_sphere_data()` for a sphere with the r grid `grid`; kwargs pass through."""
    q, iq, diq = _sphere_data()
    return invert("sphere", q, iq, diq, {"r": grid}, sld=4.0, sld_solvent=1.0, **kwargs)


class TestSpherePipeline:
    """
    With the GALAHAD solver mocked, check that invert() packages its output
    correctly for a single-parameter model (sphere): the InversionResult fields,
    the volume-weighted distribution, the shapes of theory and residuals, and
    that a (min, max, nbins) grid tuple and a prebuilt grid array give the
    same result.
    """

    def test_invert_returns_a_populated_inversion_result(self, fake_optimize):
        """
        Test that invert() returns an InversionResult and that a
        single-parameter model gets exactly one ParamDistribution, with its
        volume-weighted distribution computed. Testing mostly plumbing.
        """
        result = _invert_sphere()

        assert isinstance(result, InversionResult)
        assert len(result.distributions) == 1
        dist = result.distributions[0]
        assert isinstance(dist, ParamDistribution)
        # single-parameter models get a volume-weighted distribution
        assert dist.volume_weights is not None

    def test_invert_returns_theory_and_residuals_on_the_q_grid(self, fake_optimize):
        """
        Test that the fitted curve and residuals come back on the input q grid.
        Shapes only.
        """
        q, _, _ = _sphere_data()
        result = _invert_sphere()

        assert result.theory.shape == q.shape
        assert result.residuals.shape == q.shape

    def test_invert_treats_a_grid_tuple_and_array_identically(self, fake_optimize):
        """
        Test that a (min, max, nbins) tuple and a prebuilt array of bin
        centres produce the same grid, theory and chi2.
        """
        grid = np.linspace(10.0, 100.0, 16)

        from_tuple = _invert_sphere()
        from_array = _invert_sphere(grid=grid)

        np.testing.assert_array_equal(
            from_tuple.distributions[0].grid, from_array.distributions[0].grid
        )
        np.testing.assert_array_equal(from_tuple.theory, from_array.theory)
        assert from_tuple.chi2 == from_array.chi2


class TestSolverContract:
    """
    Test exactly what invert() passes to the GALAHAD solver (the shape of the
    Green's tensor G and the regularisation weight sigma).
    """

    def test_invert_calls_optimize_once_with_G_and_sigma(self, fake_optimize):
        """
        Test that the solver is called exactly once with the right G shape and
        sigma.
        """
        q, _, _ = _sphere_data()
        _invert_sphere(sigma=0.25)

        # once, not iteratively: the problem is convex
        assert len(fake_optimize.calls) == 1
        call = fake_optimize.calls[0]
        # G's leading axis is q; trailing axis is the r grid
        assert call["G_shape"] == (len(q), 16)
        assert call["sigma"] == 0.25

    def test_invert_passes_sigma_none_when_not_given(self, fake_optimize):
        """
        Test that omitting sigma reaches the solver as None, not as 0.0.
        None disables regularisation entirely; 0.0 would add a zero-weight
        term that changes the Jacobian structure.
        """
        _invert_sphere()
        assert fake_optimize.calls[0]["sigma"] is None


class TestParameterOrdering:
    """
    Test that multi-parameter models (cylinder, ellipsoid) return their
    distributions in ffsi's canonical parameter order whatever order the grids
    dict is given in. SasView and ffsi order these parameters differently, so
    callers must not depend on dict order.
    """

    def test_invert_returns_cylinder_params_in_canonical_order(self, fake_optimize):
        """
        Test that cylinder distributions come back as (l, r) regardless of
        the dict order. ffsi's canonical order is (l, r); sasmodels uses
        (radius, length). The grid dict deliberately has r before l to prove
        that ordering is driven by ffsi's parameter schema, not the dict.
        """
        q, iq, diq = _sphere_data()
        # grids given in the "wrong" dict order: r before l
        result = invert('cylinder', q, iq, diq,
                        {'r': (20.0, 60.0, 8), 'l': (100.0, 400.0, 8)},
                        sld=4.0, sld_solvent=1.0)

        assert [d.name for d in result.distributions] == ['l', 'r']
        for dist in result.distributions:
            np.testing.assert_array_equal(dist.weights, np.full(8, 0.125))
            # per-marginal volume weighting is undefined for multi-param models
            assert dist.volume_weights is None

    def test_invert_returns_ellipsoid_params_in_canonical_order(self, fake_optimize):
        """
        Test that ellipsoid distributions come back as (rp, re) regardless of
        dict order.
        """
        q, iq, diq = _sphere_data()
        result = invert('ellipsoid', q, iq, diq,
                        {'re': (20.0, 60.0, 8), 'rp': (30.0, 90.0, 8)},
                        sld=4.0, sld_solvent=1.0)

        assert [d.name for d in result.distributions] == ['rp', 're']
        for dist in result.distributions:
            np.testing.assert_array_equal(dist.weights, np.full(8, 0.125))
            assert dist.volume_weights is None


class TestSmearingPath:
    """
    Test that invert() builds a resolution-smeared G only when both q_calc and
    resolution_weights are given, and falls back to the unsmeared G when either
    is missing.
    """

    def _spy(self, monkeypatch):
        """Record which G-building method invert() reaches for the sphere."""
        seen = []
        real_smeared = Sphere.compute_smeared_scattering_intensity
        real_plain = Sphere.compute_scattering_intensity

        def smeared(q_calc_list, q_calc_weights, param_list, drho):
            seen.append('smeared')
            return real_smeared(q_calc_list, q_calc_weights, param_list, drho)

        def plain(q_list, param_list, drho):
            seen.append('plain')
            return real_plain(q_list, param_list, drho)

        monkeypatch.setattr(Sphere, 'compute_smeared_scattering_intensity', staticmethod(smeared))
        monkeypatch.setattr(Sphere, 'compute_scattering_intensity', staticmethod(plain))
        return seen

    def test_invert_takes_the_smeared_branch(self, fake_optimize, monkeypatch):
        """
        Test that when both q_calc and resolution_weights are given, invert()
        builds G on the extended q points (grid).
        """
        seen = self._spy(monkeypatch)
        q, _, _ = _sphere_data()
        # identity weight matrix, shape (len(q_calc), len(q))
        weights = np.eye(len(q))
        result = _invert_sphere(q_calc=q, resolution_weights=weights)

        assert 'smeared' in seen
        assert result.theory.shape == q.shape

    def test_invert_ignores_q_calc_without_resolution_weights(
        self, fake_optimize, monkeypatch
    ):
        """
        Test that a half-specified smearing request falls back to the plain path
        """
        seen = self._spy(monkeypatch)
        q, _, _ = _sphere_data()
        _invert_sphere(q_calc=q)

        assert seen == ['plain']


class TestModelResolutionErrors:
    """
    Test that model names (and SASModel classes) resolve to the right model,
    and that an unrecognised model name raises a clear ValueError before any
    computation is done.
    """

    def test_resolve_model_rejects_an_unknown_name(self):
        """
        Test that an unrecognised model name raises before any work is done
        """
        with pytest.raises(ValueError, match='core_shell_sphere'):
            _resolve_model('core_shell_sphere')

    def test_resolve_model_accepts_a_sasmodel_subclass(self):
        """
        Test that a SASModel subclass (model class) is accepted as well as
        a name string (model name).
        """
        cls, name = _resolve_model(Sphere)
        assert cls is Sphere and name == 'sphere'


# ---------------------------------- sphere on synthetic data (pinned numerics)

# set ground truth values for simple toy problem, and tolerances for test values.

# discretisation
Q = np.logspace(-3, 0, 320)
R = np.linspace(400.0, 800.0, 200)

SLD = 4.0
SLD_SOLVENT = 3.0
DRHO = SLD - SLD_SOLVENT

SIGMA = {"noiseless": 0.25, "noisy": 16.0}


def gaussian_weights(grid, centre, width):
    """
    A normalised Gaussian distribution over `grid`.
    """
    weights = np.exp(-((grid - centre) ** 2) / (2 * width**2))
    return weights / weights.sum()


# ground truth
SCALE_TRUE = 2.0
BACKGROUND_TRUE = 0.5
MEAN_R_TRUE = 500.0
W_TRUE = gaussian_weights(R, MEAN_R_TRUE, 10.0)

AVERAGE_VOLUME_TRUE = 5.24227e8
XI_TRUE = 3.81514e-13

# simulated intensities, with 20%-30% error bars
_G = Sphere.compute_scattering_intensity([Q], [R], DRHO)
I_DATA = XI_TRUE * (_G @ W_TRUE) + BACKGROUND_TRUE
I_DATA_STD = (np.random.RandomState(0).rand(Q.size) * 0.1 + 0.2) * I_DATA
I_DATA_NOISY = I_DATA + I_DATA_STD * np.random.default_rng(12345).standard_normal(
    Q.size
)

DATASETS = {"noiseless": I_DATA, "noisy": I_DATA_NOISY}

CHI2 = {"noiseless": 2.15050e-11, "noisy": 0.976485}
RESIDUAL_MAX = {"noiseless": 3.33345e-05, "noisy": 3.042781}
RESIDUAL_MIN = {"noiseless": -2.53258e-05, "noisy": -2.613871}

# relative tolerance on the recovered quantities, per dataset
PROBLEM_TOL = [("noiseless", 1e-5), ("noisy", 5e-2)]


@pytest.fixture(scope="module")
def results():
    """
    Fit the toy problem once per dataset, for the whole module.
    """
    return {
        name: invert(
            "sphere",
            Q,
            intensity,
            I_DATA_STD,
            {"r": R},
            sld=SLD,
            sld_solvent=SLD_SOLVENT,
            sigma=SIGMA[name],
        )
        for name, intensity in DATASETS.items()
    }


class TestInvert:
    """
    Run invert() end to end (real GALAHAD solver, no mocking) on a synthetic
    sphere problem with known ground truth, for noiseless and noisy data, and
    check that the recovered scale, xi, background, average volume and
    distribution match the truth, and that chi2 and residuals match their
    pinned values.
    """

    @pytest.mark.parametrize("case", DATASETS)
    def test_invert_returns_correct_drho(self, results, case):
        """
        Test that the contrast used for the fit is reported back
        """
        assert results[case].drho == pytest.approx(1.0)

    @pytest.mark.parametrize("case, rtol", PROBLEM_TOL)
    def test_invert_returns_correct_scale(self, results, case, rtol):
        """
        Test that the volume-fraction scale is recovered
        """
        assert results[case].scale == pytest.approx(SCALE_TRUE, rel=rtol)

    @pytest.mark.parametrize("case, rtol", PROBLEM_TOL)
    def test_invert_returns_correct_xi(self, results, case, rtol):
        """
        Test that xi is recovered

        `abs=0` because xi is ~1e-13, well under the default absolute
        tolerance.
        """
        assert results[case].xi == pytest.approx(XI_TRUE, rel=rtol, abs=0)

    @pytest.mark.parametrize("case, rtol", PROBLEM_TOL)
    def test_invert_returns_correct_background(self, results, case, rtol):
        """
        Test that the flat background is recovered
        """
        assert results[case].background == pytest.approx(BACKGROUND_TRUE, rel=rtol)

    @pytest.mark.parametrize("case, rtol", PROBLEM_TOL)
    def test_invert_returns_correct_average_volume(self, results, case, rtol):
        """
        Test that the average sphere volume is recovered
        """
        assert results[case].average_volume == pytest.approx(
            AVERAGE_VOLUME_TRUE, rel=rtol
        )

    @pytest.mark.parametrize("case", DATASETS)
    def test_invert_returns_valid_distribution(self, results, case):
        """
        Test that the weights satisfy the simplex constraint
        """
        weights = results[case].distribution("r").weights

        assert weights.sum() == pytest.approx(1.0, abs=1e-8)
        assert (weights >= 0).all()

    @pytest.mark.parametrize("case", DATASETS)
    def test_invert_returns_correct_chi2(self, results, case):
        """
        Test the mean squared residual against its pinned value
        """
        assert results[case].chi2 == pytest.approx(CHI2[case], abs=1e-6)

    @pytest.mark.parametrize("case", DATASETS)
    def test_invert_returns_correct_residuals(self, results, case):
        """
        Test the residual extremes against their pinned values
        """
        residuals = results[case].residuals

        assert residuals.max() == pytest.approx(RESIDUAL_MAX[case], abs=1e-6)
        assert residuals.min() == pytest.approx(RESIDUAL_MIN[case], abs=1e-6)

    @pytest.mark.parametrize("case", DATASETS)
    def test_invert_returns_correct_result_shapes(self, results, case):
        """
        Test that the fitted curve and residuals are returned on the q grid
        """
        result = results[case]

        assert result.theory.shape == Q.shape
        assert result.residuals.shape == Q.shape
        assert result.distribution("r").grid.shape == R.shape


"""
Tests for the 2D route of `ffsi.api.invert` (cylinder2d, ellipsoid2d)
against a synthetic 2D cylinder problem with known ground truth.
"""

# -------------------------------- 2D cylinder on synthetic data (pinned numerics)

# discretisation (even lengths keep q = 0 off the grid)
Q2D = np.linspace(-0.12, 0.12, 24)
L_2D = np.linspace(100.0, 140.0, 3)
R_2D = np.linspace(20.0, 30.0, 3)
THETA_2D = np.deg2rad(np.linspace(20.0, 80.0, 4))
PHI_2D = np.deg2rad(np.linspace(0.0, 180.0, 4))
GRIDS_2D = {"l": L_2D, "r": R_2D, "theta": THETA_2D, "phi": PHI_2D}

SLD_2D = 4.0
SLD_SOLVENT_2D = 1.0
SIGMA_2D = 0.25

# ground truth: every particle has L = 120, r = 25, theta = 40 deg, phi = 60 deg
W_TRUE_2D = [np.eye(3)[1], np.eye(3)[1], np.eye(4)[1], np.eye(4)[1]]
TRUE_INDEX_2D = [1, 1, 1, 1]
SCALE_TRUE_2D = 0.01
BACKGROUND_TRUE_2D = 0.001

AVERAGE_VOLUME_TRUE_2D = np.pi * 120.0 * 25.0**2
XI_TRUE_2D = scale_to_xi(SCALE_TRUE_2D, AVERAGE_VOLUME_TRUE_2D)

# simulated images, with 1% error bars
_G2D = Cylinder2D.compute_scattering_intensity(
    [Q2D, Q2D], [L_2D, R_2D, THETA_2D, PHI_2D], SLD_2D - SLD_SOLVENT_2D
)
I_2D = (
    XI_TRUE_2D * contract_tensor(_G2D, W_TRUE_2D, skip_axes=[0, 1]) + BACKGROUND_TRUE_2D
)
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
        name: invert(
            "cylinder2d",
            (Q2D, Q2D),
            intensity,
            I_2D_STD,
            GRIDS_2D,
            sld=SLD_2D,
            sld_solvent=SLD_SOLVENT_2D,
            sigma=SIGMA_2D,
        )
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
        assert results_2d[case].background == pytest.approx(
            BACKGROUND_TRUE_2D, rel=rtol
        )

    @pytest.mark.parametrize("case, rtol", PROBLEM_TOL_2D)
    def test_invert_returns_correct_average_volume(self, results_2d, case, rtol):
        """
        Test that the average cylinder volume is recovered
        """
        assert results_2d[case].average_volume == pytest.approx(
            AVERAGE_VOLUME_TRUE_2D, rel=rtol
        )

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

        assert [
            int(np.argmax(d.weights)) for d in result.distributions
        ] == TRUE_INDEX_2D

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
