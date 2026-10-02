"""
Tests for the public API (`ffsi.api.invert`).

Run with: python -m pytest tests/
"""
import os

import numpy as np
import pytest

from ffsi.api import invert, InversionResult, ParamDistribution, _resolve_model
from ffsi.models.sphere import Sphere
from ffsi.models.cylinder import Cylinder
from ffsi.models.ellipsoid import Ellipsoid
from ffsi.utils import contract_tensor, smear_tensor_1d

DATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'ffsi', 'data', 'SANS', 'observation.txt')
TRUNCATE = 285          # drop noisy high-q tail
R_MIN, R_MAX, N_BINS = 400.0, 800.0, 100
SIGMA = 0.25

# contrast is always supplied. Sphere SasView defaults: sld - sld_solvent = 1 - 6
SLD, SLD_SOLVENT = 1.0, 6.0
DRHO = SLD - SLD_SOLVENT
# a second, different contrast (drho = 1) for the contrast-invariance tests
SLD_ALT, SLD_SOLVENT_ALT = 7.0, 6.0
DRHO_ALT = SLD_ALT - SLD_SOLVENT_ALT


def load_sans():
    data = np.loadtxt(DATA_FILE)
    return data[:TRUNCATE, 0], data[:TRUNCATE, 1], data[:TRUNCATE, 2]


# -------------------------------------------------------- model resolution

class TestModelResolution:

    def test_lookup_by_name(self):
        assert _resolve_model('sphere') == (Sphere, 'sphere')
        assert _resolve_model('cylinder') == (Cylinder, 'cylinder')
        assert _resolve_model('ellipsoid') == (Ellipsoid, 'ellipsoid')

    def test_lookup_is_case_insensitive(self):
        assert _resolve_model('Sphere') == (Sphere, 'sphere')

    def test_lookup_by_class(self):
        assert _resolve_model(Sphere) == (Sphere, 'sphere')

    def test_unknown_model(self):
        with pytest.raises(ValueError, match='sphere'):
            _resolve_model('core_shell_sphere')

    def test_param_name_orders(self):
        assert _resolve_model('sphere')[0].param_names_scattering_intensity == ['r']
        assert _resolve_model('cylinder')[0].param_names_scattering_intensity == ['l', 'r']
        assert _resolve_model('ellipsoid')[0].param_names_scattering_intensity == ['rp', 're']


# ------------------------------------------------- sphere on real SANS data

@pytest.fixture(scope='module')
def sans_data():
    return load_sans()


@pytest.fixture(scope='module')
def sphere_result(sans_data):
    q, iq, diq = sans_data
    return invert('sphere', q, iq, diq, {'r': (R_MIN, R_MAX, N_BINS)},
                  sld=SLD, sld_solvent=SLD_SOLVENT, sigma=SIGMA)


class TestSphereRealData:

    def test_result_shape(self, sphere_result, sans_data):
        q = sans_data[0]
        assert isinstance(sphere_result, InversionResult)
        assert sphere_result.model == 'sphere'
        assert sphere_result.theory.shape == q.shape
        assert sphere_result.residuals.shape == q.shape
        assert np.all(np.isfinite(sphere_result.theory))

    def test_chi2(self, sphere_result, sans_data):
        # reference chi2/Npts ~ 0.85 for this dataset and grid
        assert sphere_result.chi2 < 1.5
        q, iq, diq = sans_data
        residuals = (sphere_result.theory - iq) / diq
        assert sphere_result.chi2 == pytest.approx(np.sum(residuals ** 2) / residuals.size)

    def test_background(self, sphere_result):
        # reference b ~ 0.0867
        assert sphere_result.background == pytest.approx(0.0867, abs=0.02)

    def test_distribution(self, sphere_result):
        assert len(sphere_result.distributions) == 1
        dist = sphere_result.distributions[0]
        assert isinstance(dist, ParamDistribution)
        assert dist.name == 'r'
        assert dist.grid.shape == (N_BINS,)
        assert dist.grid[0] == R_MIN and dist.grid[-1] == R_MAX
        # weights on the simplex
        assert dist.weights.min() >= -1e-10
        assert np.sum(dist.weights) == pytest.approx(1.0)
        # reference peak ~ 710 A
        assert 600 < dist.grid[np.argmax(dist.weights)] < 800

    def test_volume_weights(self, sphere_result):
        dist = sphere_result.distribution('r')
        expected = dist.weights * dist.grid ** 3
        expected /= np.sum(expected)
        assert np.allclose(dist.volume_weights, expected)

    def test_distribution_lookup(self, sphere_result):
        assert sphere_result.distribution('r') is sphere_result.distributions[0]
        with pytest.raises(KeyError):
            sphere_result.distribution('x')

    def test_average_volume(self, sphere_result):
        dist = sphere_result.distribution('r')
        expected = np.sum(dist.weights * 4 / 3 * np.pi * dist.grid ** 3)
        assert sphere_result.average_volume == pytest.approx(expected)

    def test_xi_positive(self, sphere_result):
        assert sphere_result.xi > 0
        assert np.isfinite(sphere_result.background)

    def test_grid_as_array_equivalent(self, sphere_result, sans_data):
        q, iq, diq = sans_data
        grid = np.linspace(R_MIN, R_MAX, N_BINS)
        result = invert('sphere', q, iq, diq, {'r': grid},
                        sld=SLD, sld_solvent=SLD_SOLVENT, sigma=SIGMA)
        assert result.xi == sphere_result.xi
        assert result.background == sphere_result.background
        assert np.array_equal(result.distributions[0].weights,
                              sphere_result.distributions[0].weights)

    def test_model_class_accepted(self, sans_data):
        q, iq, diq = sans_data
        result = invert(Sphere, q, iq, diq, {'r': (R_MIN, R_MAX, 20)},
                        sld=SLD, sld_solvent=SLD_SOLVENT, sigma=SIGMA)
        assert result.model == 'sphere'
        assert len(result.distributions) == 1


# ----------------------------------------------------------------------- GPU

class TestGPU:
    """Passing cupy arrays runs on the GPU but returns host numpy that matches
    the CPU fit. Skipped where CuPy is not installed."""

    def test_gpu_matches_cpu(self, sphere_result, sans_data):
        cp = pytest.importorskip('cupy')
        q, iq, diq = sans_data
        result = invert('sphere', cp.asarray(q), cp.asarray(iq), cp.asarray(diq),
                        {'r': (R_MIN, R_MAX, N_BINS)},
                        sld=SLD, sld_solvent=SLD_SOLVENT, sigma=SIGMA)
        # results come back as host numpy, not cupy
        assert isinstance(result.theory, np.ndarray)
        assert isinstance(result.distribution('r').weights, np.ndarray)
        # and agree with the CPU fit
        np.testing.assert_allclose(result.theory, sphere_result.theory, rtol=1e-6)
        np.testing.assert_allclose(result.distribution('r').weights,
                                   sphere_result.distribution('r').weights, atol=1e-8)


# ------------------------------------------------------------------ contrast

@pytest.fixture(scope='module')
def sphere_result_alt(sans_data):
    """Same fit at a different contrast (drho=1), for the invariance tests."""
    q, iq, diq = sans_data
    return invert('sphere', q, iq, diq, {'r': (R_MIN, R_MAX, N_BINS)},
                  sld=SLD_ALT, sld_solvent=SLD_SOLVENT_ALT, sigma=SIGMA)


class TestContrast:
    """
    `G` scales as `drho**2` and `xi` is free, so the contrast cannot be fitted;
    it is an input baked into `G`. The fitted distribution/curve are therefore
    contrast-invariant, `xi` absorbs `drho**2`, and `scale = xi*<V>*1e4` is the
    reported volume fraction.
    """

    def test_drho_from_slds(self, sphere_result):
        assert sphere_result.drho == DRHO
        assert sphere_result.scale > 0

    def test_fit_is_contrast_invariant(self, sphere_result, sphere_result_alt):
        """
        Changing only the contrast rescales `G` by a global factor that `xi`
        absorbs, so the fitted distribution and curve must not depend on it.
        Agreement is numerical: GALAHAD solves a problem scaled by drho**2 and
        converges to a very slightly different point.
        """
        base, alt = sphere_result, sphere_result_alt
        np.testing.assert_allclose(alt.distribution('r').weights,
                                   base.distribution('r').weights, atol=1e-8)
        np.testing.assert_allclose(alt.theory, base.theory, rtol=1e-9)
        assert alt.background == pytest.approx(base.background, rel=1e-9)
        assert alt.chi2 == pytest.approx(base.chi2, rel=1e-9)
        # xi absorbs drho**2: the xi ratio is the inverse-square of the drho ratio
        assert alt.xi / base.xi == pytest.approx((DRHO / DRHO_ALT) ** 2, rel=1e-9)

    def test_scale_no_double_counting(self, sphere_result, sphere_result_alt):
        """
        drho**2 already lives in `G`, so `scale = xi*<V>*1e4` must not apply it
        again. Since xi is proportional to 1/drho**2, `scale * drho**2` is
        contrast-independent; a mismatch means the contrast was counted twice.
        """
        assert (sphere_result.scale * DRHO ** 2
                == pytest.approx(sphere_result_alt.scale * DRHO_ALT ** 2, rel=1e-9))


# ------------------------------------------- multi-parameter models (smoke)

def synthetic_data(model_class, grids, xi_true=2.0, b_true=0.5, seed=0):
    """Small synthetic dataset with a known separable distribution."""
    rng = np.random.default_rng(seed)
    q = np.linspace(2e-3, 2e-1, 40)
    param_list = [np.linspace(lo, hi, n) for lo, hi, n in grids]
    w_list = []
    for grid in param_list:
        w = np.exp(-0.5 * ((grid - grid.mean()) / (0.15 * np.ptp(grid))) ** 2)
        w_list.append(w / np.sum(w))
    G = model_class.compute_scattering_intensity([q], param_list, 1.0)
    intensity = xi_true * np.asarray(contract_tensor(G, w_list, skip_axes=[0])) + b_true
    intensity_std = 0.05 * np.abs(intensity)
    intensity = intensity + intensity_std * rng.standard_normal(q.size)
    return q, intensity, intensity_std


class TestCylinder:

    def test_smoke(self):
        grids = [(100.0, 400.0, 8), (20.0, 60.0, 8)]  # l, r
        q, iq, diq = synthetic_data(Cylinder, grids)
        result = invert('cylinder', q, iq, diq, {'r': grids[1], 'l': grids[0]},
                        sld=SLD, sld_solvent=SLD_SOLVENT, sigma=SIGMA)
        # distributions come back in ffsi's canonical order, not dict order
        assert [d.name for d in result.distributions] == ['l', 'r']
        for dist in result.distributions:
            assert np.sum(dist.weights) == pytest.approx(1.0)
            assert dist.weights.min() >= -1e-10
            assert dist.volume_weights is None
        assert np.isfinite(result.chi2)
        assert result.average_volume > 0


class TestEllipsoid:

    def test_smoke(self):
        grids = [(150.0, 350.0, 8), (30.0, 80.0, 8)]  # rp, re
        q, iq, diq = synthetic_data(Ellipsoid, grids)
        result = invert('ellipsoid', q, iq, diq, {'rp': grids[0], 're': grids[1]},
                        sld=SLD, sld_solvent=SLD_SOLVENT, sigma=SIGMA)
        assert [d.name for d in result.distributions] == ['rp', 're']
        for dist in result.distributions:
            assert np.sum(dist.weights) == pytest.approx(1.0)
            assert dist.weights.min() >= -1e-10
            assert dist.volume_weights is None
        assert np.isfinite(result.chi2)
        assert result.average_volume > 0


# ----------------------------------------------- resolution smearing (1D)

class TestSmearing:
    """
    Passing (q_calc, resolution_weights) builds G on the extended grid and
    smears it back onto the measured q. The weights come from sasmodels
    Pinhole1D/Slit1D, matching how SasView drives the smeared fit.
    """

    GRID = {'r': (R_MIN, R_MAX, N_BINS)}

    def test_pinhole_result_shape(self, sans_data):
        from sasmodels.resolution import Pinhole1D
        q, iq, diq = sans_data
        res = Pinhole1D(q, 0.05 * q)  # 5% dQ/Q gaussian resolution
        result = invert('sphere', q, iq, diq, self.GRID,
                        sld=SLD, sld_solvent=SLD_SOLVENT, sigma=SIGMA,
                        q_calc=res.q_calc, resolution_weights=res.weight_matrix)
        assert isinstance(result, InversionResult)
        # theory/residuals live on the measured q, not the extended q_calc
        assert result.theory.shape == q.shape
        assert result.residuals.shape == q.shape
        assert np.all(np.isfinite(result.theory))
        assert np.isfinite(result.chi2)

    def test_pinhole_changes_the_fit(self, sphere_result, sans_data):
        from sasmodels.resolution import Pinhole1D
        q, iq, diq = sans_data
        res = Pinhole1D(q, 0.05 * q)
        smeared = invert('sphere', q, iq, diq, self.GRID,
                         sld=SLD, sld_solvent=SLD_SOLVENT, sigma=SIGMA,
                         q_calc=res.q_calc, resolution_weights=res.weight_matrix)
        # smearing must actually change the fitted curve and distribution
        assert not np.allclose(smeared.theory, sphere_result.theory)
        assert not np.allclose(smeared.distribution('r').weights,
                               sphere_result.distribution('r').weights)

    def test_pinhole_matches_manual_forward(self, sans_data):
        from sasmodels.resolution import Pinhole1D
        q, iq, diq = sans_data
        res = Pinhole1D(q, 0.05 * q)
        result = invert('sphere', q, iq, diq, self.GRID,
                        sld=SLD, sld_solvent=SLD_SOLVENT, sigma=SIGMA,
                        q_calc=res.q_calc, resolution_weights=res.weight_matrix)
        # reconstruct theory = xi * smear(G).w + background from the fit outputs
        grid = np.linspace(R_MIN, R_MAX, N_BINS)
        weights = result.distribution('r').weights
        G = Sphere.compute_scattering_intensity([res.q_calc], [grid], DRHO)
        G_smeared = smear_tensor_1d(G, res.weight_matrix)
        theory = result.xi * np.asarray(contract_tensor(G_smeared, [weights], skip_axes=[0])) \
            + result.background
        np.testing.assert_allclose(result.theory, theory, rtol=1e-9)

    def test_slit_smoke(self, sphere_result, sans_data):
        from sasmodels.resolution import Slit1D
        q, iq, diq = sans_data
        res = Slit1D(q, q_length=0.03, q_width=0.0)
        result = invert('sphere', q, iq, diq, self.GRID,
                        sld=SLD, sld_solvent=SLD_SOLVENT, sigma=SIGMA,
                        q_calc=res.q_calc, resolution_weights=res.weight_matrix)
        assert result.theory.shape == q.shape
        assert np.all(np.isfinite(result.theory))
        assert np.isfinite(result.chi2)
        assert not np.allclose(result.theory, sphere_result.theory)

    def test_partial_smearing_args_ignored(self, sphere_result, sans_data):
        """q_calc without resolution_weights (or vice versa) is the unsmeared path."""
        q, iq, diq = sans_data
        only_qcalc = invert('sphere', q, iq, diq, self.GRID,
                            sld=SLD, sld_solvent=SLD_SOLVENT, sigma=SIGMA,
                            q_calc=q)
        np.testing.assert_allclose(only_qcalc.theory, sphere_result.theory, rtol=1e-9)
