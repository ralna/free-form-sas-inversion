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
from ffsi.utils import scale_to_xi


@pytest.fixture(autouse=True)
def force_cpu(monkeypatch):
    """Run on numpy so G is a host array."""
    monkeypatch.setattr("ffsi.array_module.CUPY_INSTALLED", False)


SLD = 4.0
SLD_SOLVENT = 1.0
DRHO = SLD - SLD_SOLVENT

# q range over which ffsi's fixed 76-point quadrature is accurate for the
# sizes below (see TestQuadratureLimit for where it is not)
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
class TestQuadratureLimit:
    """
    Document a known limitation: the 1D cylinder and ellipsoid integrate over
    orientation with a fixed 76-point Gauss-Legendre rule, which is not
    accurate once q times the largest dimension reaches a few hundred.
    This matches sasmodels up to 1.0.x (SasView up to 6.1), which used the
    same fixed rule (lib/gauss76.c). Since sasmodels 1.1.0 (bundled with
    SasView 6.2.0) lib/adaptive.c picks 20, 76, 500 or 5000 points from
    q * r_max (thresholds 10, 100, 750), which stays accurate.

    These are strict xfails, so they will start failing (and should be turned
    into ordinary tests) once ffsi's quadrature is made adaptive.
    """

    @pytest.mark.xfail(strict=True, reason="fixed 76-point quadrature at large q*L")
    def test_long_cylinder_matches_sasmodels_at_high_q(self):
        q = np.logspace(-3, 0, 60)
        expected = sasmodels_intensity("cylinder", [q], length=1000.0, radius=10.0)
        actual = ffsi_intensity(Cylinder, [q], [1000.0, 10.0])

        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    @pytest.mark.xfail(strict=True, reason="fixed 76-point quadrature at large q*r")
    def test_large_oblate_ellipsoid_matches_sasmodels_at_high_q(self):
        q = np.logspace(-3, 0, 60)
        expected = sasmodels_intensity(
            "ellipsoid", [q], radius_polar=30.0, radius_equatorial=800.0
        )
        actual = ffsi_intensity(Ellipsoid, [q], [30.0, 800.0])

        np.testing.assert_allclose(actual, expected, rtol=1e-6)


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
