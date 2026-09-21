"""
Mock-based plumbing tests for `ffsi.api.invert`.
"""
import numpy as np
import pytest

from ffsi.api import ParamDistribution, InversionResult, _resolve_model, invert
from ffsi.models.cylinder import Cylinder
from ffsi.models.ellipsoid import Ellipsoid
from ffsi.models.sphere import Sphere
from ffsi.utils import xi_to_scale

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
    q = np.linspace(0.005, 0.2, n)
    iq = np.linspace(10.0, 1.0, n)
    diq = 0.1 * iq
    return q, iq, diq


class TestSpherePipeline:
    """The single-parameter (sphere) path end to end, solver mocked."""

    def test_result_structure(self, fake_optimize):
        q, iq, diq = _sphere_data()
        result = invert('sphere', q, iq, diq, {'r': (10.0, 100.0, 16)},
                        sld=4.0, sld_solvent=1.0)

        assert isinstance(result, InversionResult)
        assert result.model == 'sphere'
        assert result.xi == FAKE_XI
        assert result.background == FAKE_BACKGROUND
        assert result.drho == pytest.approx(3.0)  # sld - sld_solvent

        assert len(result.distributions) == 1
        dist = result.distributions[0]
        assert isinstance(dist, ParamDistribution)
        assert dist.name == 'r'
        assert len(dist.grid) == 16
        assert dist.grid[0] == pytest.approx(10.0)
        assert dist.grid[-1] == pytest.approx(100.0)
        assert dist.weights.sum() == pytest.approx(1.0)
        # single-parameter models get a volume-weighted distribution
        assert dist.volume_weights is not None
        assert dist.volume_weights.sum() == pytest.approx(1.0)

    def test_theory_residuals_chi2(self, fake_optimize):
        q, iq, diq = _sphere_data()
        result = invert('sphere', q, iq, diq, {'r': (10.0, 100.0, 16)},
                        sld=4.0, sld_solvent=1.0)

        # does the fit match the curve
        assert result.theory.shape == q.shape
        np.testing.assert_allclose(result.residuals, (result.theory - iq) / diq)
        assert result.chi2 == pytest.approx(
            np.sum(result.residuals ** 2) / result.residuals.size)

    def test_scale_from_average_volume(self, fake_optimize):
        q, iq, diq = _sphere_data()
        result = invert('sphere', q, iq, diq, {'r': (10.0, 100.0, 16)},
                        sld=4.0, sld_solvent=1.0)

        assert result.scale == pytest.approx(
            xi_to_scale(result.xi, result.average_volume))

    # check whether grids form correctly
    def test_grid_triple_and_array_equivalent(self, fake_optimize):
        q, iq, diq = _sphere_data()
        grid = np.linspace(10.0, 100.0, 16)

        from_triple = invert('sphere', q, iq, diq, {'r': (10.0, 100.0, 16)},
                             sld=4.0, sld_solvent=1.0)
        from_array = invert('sphere', q, iq, diq, {'r': grid},
                            sld=4.0, sld_solvent=1.0)

        np.testing.assert_allclose(from_triple.distributions[0].grid,
                                   from_array.distributions[0].grid)
        np.testing.assert_allclose(from_triple.theory, from_array.theory)


class TestSolverContract:
    """What `invert` hands the solver, and that it calls it exactly once."""
    def test_optimize_called_once_with_G_and_sigma(self, fake_optimize):
        q, iq, diq = _sphere_data()
        invert('sphere', q, iq, diq, {'r': (10.0, 100.0, 16)},
               sld=4.0, sld_solvent=1.0, sigma=0.25)

        assert len(fake_optimize.calls) == 1 # convex
        call = fake_optimize.calls[0]
        # G's leading axis is q; trailing axis is the r grid
        assert call["G_shape"] == (len(q), 16)
        assert call["sigma"] == 0.25

    def test_sigma_defaults_to_none(self, fake_optimize):
        q, iq, diq = _sphere_data()
        invert('sphere', q, iq, diq, {'r': (10.0, 100.0, 16)},
               sld=4.0, sld_solvent=1.0)
        assert fake_optimize.calls[0]["sigma"] is None


class TestParameterOrdering:
    """Multi-parameter models return distributions in ffsi's canonical order."""
    def test_cylinder_canonical_order(self, fake_optimize):
        q, iq, diq = _sphere_data()
        # grids given in the "wrong" dict order: r before l
        result = invert('cylinder', q, iq, diq,
                        {'r': (20.0, 60.0, 8), 'l': (100.0, 400.0, 8)},
                        sld=4.0, sld_solvent=1.0)

        assert [d.name for d in result.distributions] == ['l', 'r']
        for dist in result.distributions:
            assert dist.weights.sum() == pytest.approx(1.0)
            # per-marginal volume weighting is undefined for multi-param models
            assert dist.volume_weights is None

    def test_ellipsoid_canonical_order(self, fake_optimize):
        q, iq, diq = _sphere_data()
        result = invert('ellipsoid', q, iq, diq,
                        {'re': (20.0, 60.0, 8), 'rp': (30.0, 90.0, 8)},
                        sld=4.0, sld_solvent=1.0)

        assert [d.name for d in result.distributions] == ['rp', 're']
        for dist in result.distributions:
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

    # identity weight matrix, shape (len(q_calc), len(q)); q_calc == q
    def test_smearing_branch_taken(self, fake_optimize, monkeypatch):
        seen = self._spy(monkeypatch)
        q, iq, diq = _sphere_data()
        weights = np.eye(len(q))
        result = invert('sphere', q, iq, diq, {'r': (10.0, 100.0, 16)},
                        sld=4.0, sld_solvent=1.0,
                        q_calc=q, resolution_weights=weights)

        assert 'smeared' in seen
        assert result.theory.shape == q.shape

    # only q_calc, no weights -> unsmeared path
    def test_partial_smearing_args_ignored(self, fake_optimize, monkeypatch):
        seen = self._spy(monkeypatch)
        q, iq, diq = _sphere_data()
        invert('sphere', q, iq, diq, {'r': (10.0, 100.0, 16)},
               sld=4.0, sld_solvent=1.0, q_calc=q)

        assert seen == ['plain']


class TestModelResolutionErrors:
    """Paths that never reach the solver."""

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match='core_shell_sphere'):
            _resolve_model('core_shell_sphere')

    def test_model_class_accepted(self):
        cls, name = _resolve_model(Sphere)
        assert cls is Sphere and name == 'sphere'
