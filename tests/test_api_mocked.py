"""
Mock-based plumbing tests for `ffsi.api.invert`.
"""
import numpy as np
import pytest

from ffsi.api import ParamDistribution, InversionResult, _resolve_model, invert
from ffsi.models.sphere import Sphere

# canned solver outputs, asserted throughout
FAKE_XI = 1.5
FAKE_BACKGROUND = 0.1


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
        return FAKE_XI, FAKE_BACKGROUND, w_list


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


class TestSpherePipeline:
    """The single-parameter (sphere) path end to end, solver mocked."""

    def test_invert_returns_a_populated_inversion_result(self, fake_optimize):
        """
        Test that the solver's output is packaged into an InversionResult
        Testing mostly plumbing
        """
        q, iq, diq = _sphere_data()
        result = invert('sphere', q, iq, diq, {'r': (10.0, 100.0, 16)},
                        sld=4.0, sld_solvent=1.0)

        assert isinstance(result, InversionResult)
        assert result.model == 'sphere'
        assert result.xi == FAKE_XI
        assert result.background == FAKE_BACKGROUND
        assert result.drho == 3.0

        assert len(result.distributions) == 1
        dist = result.distributions[0]
        assert isinstance(dist, ParamDistribution)
        assert dist.name == 'r'
        assert len(dist.grid) == 16
        assert dist.grid[0] == 10.0
        assert dist.grid[-1] == 100.0
        # single-parameter models get a volume-weighted distribution
        assert dist.volume_weights is not None
        # not a restatement of api.py: under the mock's uniform weights the
        # 4/3*pi cancels out of its `weighted / sum(weighted)`, so the
        # expected value collapses to the closed form r**3 / sum(r**3)
        np.testing.assert_allclose(
            dist.volume_weights, dist.grid**3 / (dist.grid**3).sum(), rtol=1e-15
        )

    def test_invert_returns_theory_and_residuals_on_the_q_grid(self, fake_optimize):
        """
        Test that the fitted curve and residuals come back on the input q.
        Shapes only, Values tested in test_galahad.py
        """
        q, iq, diq = _sphere_data()
        result = invert('sphere', q, iq, diq, {'r': (10.0, 100.0, 16)},
                        sld=4.0, sld_solvent=1.0)

        assert result.theory.shape == q.shape
        assert result.residuals.shape == q.shape

    def test_invert_treats_a_grid_triple_and_array_identically(self, fake_optimize):
        """
        Test that a (min, max, nbins) triple and a prebuilt array agree
        """
        q, iq, diq = _sphere_data()
        grid = np.linspace(10.0, 100.0, 16)

        from_triple = invert('sphere', q, iq, diq, {'r': (10.0, 100.0, 16)},
                             sld=4.0, sld_solvent=1.0)
        from_array = invert('sphere', q, iq, diq, {'r': grid},
                            sld=4.0, sld_solvent=1.0)

        np.testing.assert_array_equal(
            from_triple.distributions[0].grid, from_array.distributions[0].grid
        )
        np.testing.assert_array_equal(from_triple.theory, from_array.theory)
        assert from_triple.chi2 == from_array.chi2


class TestSolverContract:
    """What `invert` hands the solver, and that it calls it exactly once."""

    def test_invert_calls_optimize_once_with_G_and_sigma(self, fake_optimize):
        """
        Test that the solver sees one call, with the right G and sigma
        """
        q, iq, diq = _sphere_data()
        invert('sphere', q, iq, diq, {'r': (10.0, 100.0, 16)},
               sld=4.0, sld_solvent=1.0, sigma=0.25)

        # once, not iteratively: the problem is convex
        assert len(fake_optimize.calls) == 1
        call = fake_optimize.calls[0]
        # G's leading axis is q; trailing axis is the r grid
        assert call["G_shape"] == (len(q), 16)
        assert call["sigma"] == 0.25

    def test_invert_passes_sigma_none_when_not_given(self, fake_optimize):
        """
        Test that omitting sigma reaches the solver as None, not as 0.0
        """
        q, iq, diq = _sphere_data()
        invert('sphere', q, iq, diq, {'r': (10.0, 100.0, 16)},
               sld=4.0, sld_solvent=1.0)
        assert fake_optimize.calls[0]["sigma"] is None


class TestParameterOrdering:
    """Multi-parameter models return distributions in ffsi's canonical order."""

    def test_invert_returns_cylinder_params_in_canonical_order(self, fake_optimize):
        """
        Test that cylinder distributions come back as (l, r)
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
        Test that ellipsoid distributions come back as (rp, re)
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
    """The smeared vs unsmeared branch selection in invert()."""

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
        Test that q_calc plus resolution_weights builds G on the extended grid
        """
        seen = self._spy(monkeypatch)
        q, iq, diq = _sphere_data()
        # identity weight matrix, shape (len(q_calc), len(q))
        weights = np.eye(len(q))
        result = invert('sphere', q, iq, diq, {'r': (10.0, 100.0, 16)},
                        sld=4.0, sld_solvent=1.0,
                        q_calc=q, resolution_weights=weights)

        assert 'smeared' in seen
        assert result.theory.shape == q.shape

    def test_invert_ignores_q_calc_without_resolution_weights(
        self, fake_optimize, monkeypatch
    ):
        """
        Test that a half-specified smearing request falls back to the plain path
        """
        seen = self._spy(monkeypatch)
        q, iq, diq = _sphere_data()
        invert('sphere', q, iq, diq, {'r': (10.0, 100.0, 16)},
               sld=4.0, sld_solvent=1.0, q_calc=q)

        assert seen == ['plain']


class TestModelResolutionErrors:
    """Paths that never reach the solver."""

    def test_resolve_model_rejects_an_unknown_name(self):
        """
        Test that an unrecognised model name raises before any work is done
        """
        with pytest.raises(ValueError, match='core_shell_sphere'):
            _resolve_model('core_shell_sphere')

    def test_resolve_model_accepts_a_sasmodel_subclass(self):
        """
        Test that a model class is accepted as well as its name
        """
        cls, name = _resolve_model(Sphere)
        assert cls is Sphere and name == 'sphere'
