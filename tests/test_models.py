"""
Tests for the scattering models in `ffsi.models`: agreement with the SasView
reference implementation (sasmodels), and consistency between each 2D
(oriented) model and its 1D (orientation-averaged) counterpart.
"""
import os

import numpy as np
import pytest

from ffsi.models.cylinder import Cylinder
from ffsi.models.cylinder2d import Cylinder2D
from ffsi.models.ellipsoid import Ellipsoid
from ffsi.models.ellipsoid2d import Ellipsoid2D
from ffsi.models.sphere import Sphere
from ffsi.models.utils import gauss_legendre
from ffsi.utils import scale_to_xi


@pytest.fixture(autouse=True)
def force_cpu(monkeypatch):
    """Run on numpy so G is a host array."""
    monkeypatch.setattr("ffsi.array_module.CUPY_INSTALLED", False)


SLD = 4.0
SLD_SOLVENT = 1.0
DRHO = SLD - SLD_SOLVENT

# q range for the model comparisons (TestAdaptiveQuadrature covers larger q*r)
Q_1D = np.logspace(-3, np.log10(0.5), 60)
QX_2D = np.linspace(-0.2, 0.2, 16)  # even lengths keep q = 0 off the grid
QY_2D = np.linspace(-0.15, 0.15, 12)


# ------------------------------------------------------------- sasmodels

# use the compiled C kernels; GitHub runners have no OpenCL device
os.environ.setdefault("SAS_OPENCL", "none")
try:
    import sasmodels.core as sasmodels_core
    from sasmodels.direct_model import call_kernel
except ImportError:
    sasmodels_core = None

needs_sasmodels = pytest.mark.skipif(
    sasmodels_core is None, reason="sasmodels is not installed"
)


def sasmodels_intensity(name, q_list, **pars):
    """
    I(q) from sasmodels with scale 1 and no background.

    `q_list` is [q] for 1D or [qx, qy] (flattened) for 2D; angles in degrees.
    """
    kernel = sasmodels_core.load_model(name, dtype="double").make_kernel(q_list)
    pars = dict(scale=1.0, background=0.0, sld=SLD, sld_solvent=SLD_SOLVENT, **pars)
    return call_kernel(kernel, pars)


def ffsi_intensity(model, q_list, param_values):
    """
    I(q) from ffsi for a single particle (one bin per parameter) at
    sasmodels' scale = 1, i.e. xi * G with xi = 1e-4 / V.
    """
    param_list = [np.array([v]) for v in param_values]
    G = model.compute_scattering_intensity(q_list, param_list, DRHO)
    n_size = len(model.param_names_average_volume)
    volume = model.compute_volume(param_list[:n_size]).item()
    return scale_to_xi(1.0, volume) * G.reshape(G.shape[: len(q_list)])


# (ffsi model, sasmodels name, ffsi parameter values, sasmodels parameter names)
CASES_1D = {
    "sphere-small": (Sphere, "sphere", [20.0], ["radius"]),
    "sphere-large": (Sphere, "sphere", [500.0], ["radius"]),
    "cylinder-rod": (Cylinder, "cylinder", [200.0, 20.0], ["length", "radius"]),
    "cylinder-disc": (Cylinder, "cylinder", [20.0, 100.0], ["length", "radius"]),
    "ellipsoid-prolate": (
        Ellipsoid, "ellipsoid", [80.0, 30.0], ["radius_polar", "radius_equatorial"]
    ),
    "ellipsoid-oblate": (
        Ellipsoid, "ellipsoid", [30.0, 80.0], ["radius_polar", "radius_equatorial"]
    ),
}

CASES_2D = {
    "cylinder2d": (Cylinder2D, "cylinder", [200.0, 20.0], ["length", "radius"]),
    "ellipsoid2d": (
        Ellipsoid2D, "ellipsoid", [80.0, 30.0], ["radius_polar", "radius_equatorial"]
    ),
}

# (theta, phi) in degrees
ORIENTATIONS = [(35.0, 70.0), (80.0, 200.0), (10.0, 300.0)]


@needs_sasmodels
class TestAgainstSasmodels:
    """
    Test that each ffsi model gives the same absolute intensity as the
    sasmodels implementation of the same SasView model, once xi is
    converted from sasmodels' volume-fraction scale.
    """

    @pytest.mark.parametrize("case", CASES_1D)
    def test_1d_model_matches_sasmodels(self, case):
        """
        Test 1D models over a range of sizes and aspect ratios
        """
        model, name, values, names = CASES_1D[case]

        expected = sasmodels_intensity(name, [Q_1D], **dict(zip(names, values)))
        actual = ffsi_intensity(model, [Q_1D], values)

        np.testing.assert_allclose(actual, expected, rtol=1e-8)

    @pytest.mark.parametrize("theta, phi", ORIENTATIONS)
    @pytest.mark.parametrize("case", CASES_2D)
    def test_2d_model_matches_sasmodels(self, case, theta, phi):
        """
        Test 2D models on a (qx, qy) grid at several orientations; ffsi takes
        angles in radians, sasmodels in degrees
        """
        model, name, values, names = CASES_2D[case]
        QX, QY = np.meshgrid(QX_2D, QY_2D, indexing="ij")

        expected = sasmodels_intensity(
            name,
            [QX.ravel(), QY.ravel()],
            theta=theta,
            phi=phi,
            **dict(zip(names, values)),
        ).reshape(QX.shape)
        actual = ffsi_intensity(
            model, [QX_2D, QY_2D], [*values, np.deg2rad(theta), np.deg2rad(phi)]
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-8)


@needs_sasmodels
class TestAdaptiveQuadrature:
    """
    Test that the 1D cylinder and ellipsoid, which integrate over orientation
    with SasView's adaptive Gauss-Legendre rule (20, 76, 500 or 5000 points
    chosen from q * r_max), match sasmodels where the integrand oscillates
    fastest and on either side of each change in the number of points.
    """

    def test_long_cylinder_matches_sasmodels_at_high_q(self):
        """
        Test q * L/2 up to 500, where a fixed 76 point rule is off by up to 6%
        """
        q = np.logspace(-3, 0, 60)
        expected = sasmodels_intensity("cylinder", [q], length=1000.0, radius=10.0)
        actual = ffsi_intensity(Cylinder, [q], [1000.0, 10.0])

        np.testing.assert_allclose(actual, expected, rtol=1e-10)

    def test_large_oblate_ellipsoid_matches_sasmodels_at_high_q(self):
        """
        Test q * re up to 800, which reaches the 5000 point rule
        """
        q = np.logspace(-3, 0, 60)
        expected = sasmodels_intensity(
            "ellipsoid", [q], radius_polar=30.0, radius_equatorial=800.0
        )
        actual = ffsi_intensity(Ellipsoid, [q], [30.0, 800.0])

        np.testing.assert_allclose(actual, expected, rtol=1e-10)

    @pytest.mark.parametrize("case", ["cylinder", "ellipsoid"])
    def test_models_match_sasmodels_either_side_of_each_threshold(self, case):
        """
        Test q * r_max just below and above 10, 100 and 750, and at 2000,
        so that each rule is used and the switch points agree with sasmodels
        """
        r_max = 500.0
        q = np.array(
            [t * f / r_max for t in (10, 100, 750) for f in (1 - 1e-9, 1 + 1e-9)]
            + [2000 / r_max]
        )
        if case == "cylinder":
            expected = sasmodels_intensity("cylinder", [q], length=2 * r_max, radius=20.0)
            actual = ffsi_intensity(Cylinder, [q], [2 * r_max, 20.0])
        else:
            expected = sasmodels_intensity(
                "ellipsoid", [q], radius_polar=r_max, radius_equatorial=30.0
            )
            actual = ffsi_intensity(Ellipsoid, [q], [r_max, 30.0])

        np.testing.assert_allclose(actual, expected, rtol=1e-10)


class TestGaussLegendre:
    """
    Test the adaptive Gauss-Legendre generator in `ffsi.models.utils`: the
    rule chosen for each q * r_max, the exactness of each rule, and that its
    blocks of points add up to the full rule.
    """

    # one entry in each range, as (q * r_max, number of points)
    RANGES = [(5.0, 20), (50.0, 76), (500.0, 500), (5000.0, 5000)]

    def _rules(self, qr_max, **kwargs):
        """
        The full rule used for each entry of qr_max, as {entry: (z, w)},
        joining the blocks that gauss_legendre yields for it.
        """
        rules = {}
        for index, z, w in gauss_legendre(np, qr_max, **kwargs):
            for entry in zip(*(i.tolist() for i in index)):
                z_all, w_all = rules.get(entry, ([], []))
                rules[entry] = (z_all + z.tolist(), w_all + w.tolist())
        return {e: (np.array(z), np.array(w)) for e, (z, w) in rules.items()}

    def test_gauss_legendre_chooses_the_number_of_points_from_qr_max(self):
        qr_max = np.array([qr for qr, _ in self.RANGES])
        rules = self._rules(qr_max)

        assert sorted(rules) == [(i,) for i in range(len(self.RANGES))]
        for i, (_, n) in enumerate(self.RANGES):
            assert rules[(i,)][0].size == n

    @pytest.mark.parametrize("qr, n", RANGES)
    def test_each_rule_integrates_polynomials_exactly(self, qr, n):
        """
        Test that the n point rule integrates x^k exactly on [-1,1] for
        k <= 50 (the rules are exact up to degree 2n-1 >= 39, so only
        k < 2n is checked). The tolerance allows for SasView's tables being
        printed to 16 significant figures, which limits the 500 and 5000
        point rules to relative errors of about 1e-12 and 1e-11.
        """
        z, w = self._rules(np.array([qr]))[(0,)]

        assert np.all(np.diff(z) > 0)
        for k in range(min(51, 2 * n)):
            exact = 2 / (k + 1) if k % 2 == 0 else 0.0
            assert w @ z**k == pytest.approx(exact, rel=1e-10, abs=1e-13)

    def test_blocks_add_up_to_the_full_rule(self):
        """
        Test that limiting the block size (here 4 points, as 2^24 elements
        are shared by 2^22 elements per point) splits the 5000 point rule
        into blocks that join back into the unsplit rule
        """
        qr_max = np.array([5000.0])
        blocks = [
            z for _, z, _ in gauss_legendre(np, qr_max, elements_per_point=2**22)
        ]
        z_split, w_split = self._rules(qr_max, elements_per_point=2**22)[(0,)]
        z_full, w_full = self._rules(qr_max)[(0,)]

        assert len(blocks) == 1250
        assert max(b.size for b in blocks) == 4
        np.testing.assert_array_equal(z_split, z_full)
        np.testing.assert_array_equal(w_split, w_full)

    def test_index_selects_entries_of_a_multidimensional_qr_max(self):
        """
        Test that the index is a nonzero-style tuple with one array per axis
        """
        qr_max = np.array([[[5.0, 50.0]], [[500.0, 5.0]]])  # shape (2, 1, 2)
        rules = self._rules(qr_max)

        assert {e: rules[e][0].size for e in rules} == {
            (0, 0, 0): 20, (0, 0, 1): 76, (1, 0, 0): 500, (1, 0, 1): 20,
        }


# --------------------------------------------------- 2D vs 1D consistency

# orientations spread uniformly over the sphere: Gauss-Legendre in
# cos(theta) and the midpoint rule in phi (spectrally accurate for periodic
# integrands, and never puts the particle axis exactly perpendicular to q)
N_ORIENT = 128
_U, _WU = np.polynomial.legendre.leggauss(N_ORIENT)
THETA_AVG = np.arccos(_U)
PHI_AVG = 2 * np.pi * (np.arange(N_ORIENT) + 0.5) / N_ORIENT

# (2D model, 1D model, size parameters)
PAIRS = {
    "cylinder": (Cylinder2D, Cylinder, [200.0, 20.0]),
    "ellipsoid-prolate": (Ellipsoid2D, Ellipsoid, [80.0, 30.0]),
    "ellipsoid-oblate": (Ellipsoid2D, Ellipsoid, [30.0, 80.0]),
}

# direction of q in the detector plane, in radians from the qx axis
DIRECTIONS = [0.0, 0.7, np.pi / 2]


class TestOrientationAverage:
    """
    Test that each 2D model, averaged over uniformly distributed particle
    orientations, reproduces the 1D model. This checks the 2D coordinate
    transformation (theta, phi) independently of sasmodels.
    """

    @pytest.mark.parametrize("direction", DIRECTIONS)
    @pytest.mark.parametrize("pair", PAIRS)
    def test_orientation_averaged_2d_model_matches_1d_model(self, pair, direction):
        """
        Test along several directions of q, which must all give the same
        result because the orientation average is isotropic
        """
        model_2d, model_1d, sizes = PAIRS[pair]
        q = np.logspace(-3, np.log10(0.5), 20)
        size_list = [np.array([s]) for s in sizes]

        expected = model_1d.compute_scattering_intensity([q], size_list, DRHO)[:, 0, 0]

        actual = np.empty_like(q)
        for i, qi in enumerate(q):
            qx = np.array([qi * np.cos(direction)])
            qy = np.array([qi * np.sin(direction)])
            G = model_2d.compute_scattering_intensity(
                [qx, qy], [*size_list, THETA_AVG, PHI_AVG], DRHO
            )[0, 0, 0, 0]
            actual[i] = (_WU / 2) @ G.mean(axis=1)

        np.testing.assert_allclose(actual, expected, rtol=1e-7)


# ---------------------------------------------------------- smearing hook


class TestSmearedScatteringIntensity:
    """
    Test that compute_smeared_scattering_intensity dispatches to the 1D or 2D
    smearing routine according to the model, so that trivial smearing leaves G
    unchanged.
    """

    def test_1d_model_with_identity_weights_matches_unsmeared(self):
        size_list = [np.array([100.0, 200.0]), np.array([10.0, 20.0, 30.0])]

        plain = Cylinder.compute_scattering_intensity([Q_1D], size_list, DRHO)
        smeared = Cylinder.compute_smeared_scattering_intensity(
            [Q_1D], np.eye(len(Q_1D)), size_list, DRHO
        )

        np.testing.assert_allclose(smeared, plain, rtol=1e-15)

    def test_2d_model_with_a_single_bin_matches_unsmeared(self):
        param_list = [
            np.array([200.0]),
            np.array([20.0]),
            np.deg2rad([30.0, 60.0]),
            np.deg2rad([0.0, 90.0]),
        ]

        plain = Cylinder2D.compute_scattering_intensity(
            [QX_2D, QY_2D], param_list, DRHO
        )
        smeared = Cylinder2D.compute_smeared_scattering_intensity(
            [QX_2D, QY_2D], np.array([1.0]), param_list, DRHO
        )

        np.testing.assert_allclose(smeared, plain, rtol=1e-15)
