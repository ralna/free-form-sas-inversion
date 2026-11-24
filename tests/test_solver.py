"""
Tests for the GALAHAD SNLS solver interface `ffsi.optimize_galahad.optimize`,
called directly (not through `ffsi.api.invert`) on small synthetic problems
with known ground truth.
"""
import numpy as np
import pytest

from ffsi.models.cylinder import Cylinder
from ffsi.models.sphere import Sphere
from ffsi.optimize_galahad import optimize
from ffsi.utils import contract_tensor


@pytest.fixture(autouse=True)
def force_cpu(monkeypatch):
    """Run on numpy so G is a host array and results are deterministic."""
    monkeypatch.setattr("ffsi.array_module.CUPY_INSTALLED", False)


def gaussian_weights(grid, centre, width):
    """A normalised Gaussian distribution over `grid`."""
    weights = np.exp(-((grid - centre) ** 2) / (2 * width**2))
    return weights / weights.sum()


# q up to 1/A so that the high-q oscillations pin down the small radii (the
# data are generated with the same G, so model accuracy at high q is moot)
Q = np.logspace(-3, 0, 120)
XI_TRUE = 3e-9
B_TRUE = 0.05


def make_problem(model, grids, weights):
    """Noiseless I(q) = xi G w + b with 5% error bars."""
    G = model.compute_scattering_intensity([Q], grids, 1.0)
    intensity = XI_TRUE * contract_tensor(G, weights, skip_axes=[0]) + B_TRUE
    return G, intensity, 0.05 * intensity


_R_SPHERE = np.linspace(20.0, 80.0, 12)
_L_CYL = np.linspace(200.0, 600.0, 10)
_R_CYL = np.linspace(5.0, 35.0, 10)

# one simplex constraint (sphere) and two (cylinder)
PROBLEMS = {
    "sphere": (Sphere, [_R_SPHERE], [gaussian_weights(_R_SPHERE, 50.0, 8.0)]),
    "cylinder": (
        Cylinder,
        [_L_CYL, _R_CYL],
        [gaussian_weights(_L_CYL, 350.0, 60.0), gaussian_weights(_R_CYL, 20.0, 5.0)],
    ),
}


@pytest.fixture(scope="module", params=PROBLEMS)
def problem(request):
    """G, data, ground-truth weights and the unregularised solution."""
    model, grids, weights = PROBLEMS[request.param]
    G, intensity, intensity_std = make_problem(model, grids, weights)
    solution = optimize(G, intensity, intensity_std)
    return {
        "G": G,
        "I": intensity,
        "I_std": intensity_std,
        "w_true": weights,
        "unregularised": solution,
    }


def objective(problem, xi, b, w_list, sigma):
    """
    The regularised least-squares objective, computed independently of the
    solver: (chi2, roughness, chi2 + sigma^2 * roughness) where chi2 is the
    sum of squared normalised residuals and roughness is sum ||diff(w)||^2.
    """
    model = xi * contract_tensor(problem["G"], w_list, skip_axes=[0]) + b
    chi2 = np.sum(((model - problem["I"]) / problem["I_std"]) ** 2)
    roughness = sum(np.sum(np.diff(w) ** 2) for w in w_list)
    return chi2, roughness, chi2 + sigma**2 * roughness


class TestRecovery:
    """
    Test that optimize() recovers xi, b and every weight vector from
    noiseless data, for one and two simplex constraints.
    """

    def test_optimize_recovers_xi_and_background(self, problem):
        xi, b, _ = problem["unregularised"]

        assert xi == pytest.approx(XI_TRUE, rel=1e-6, abs=0)
        assert b == pytest.approx(B_TRUE, rel=1e-6)

    def test_optimize_recovers_the_distributions(self, problem):
        _, _, w_list = problem["unregularised"]

        assert len(w_list) == len(problem["w_true"])
        for w, w_true in zip(w_list, problem["w_true"]):
            np.testing.assert_allclose(w, w_true, atol=1e-6)

    def test_optimize_returns_points_on_the_simplex(self, problem):
        _, _, w_list = problem["unregularised"]

        for w in w_list:
            assert w.sum() == pytest.approx(1.0, abs=1e-10)
            assert (w >= -1e-12).all()

    def test_optimize_fits_noiseless_data(self, problem):
        xi, b, w_list = problem["unregularised"]
        chi2, _, _ = objective(problem, xi, b, w_list, sigma=0.0)

        # ffsi stops once ||r|| < 1e-6
        assert chi2 < 1e-12


@pytest.fixture(scope="module", params=[1.0, 10.0])
def regularised(request, problem):
    """A regularised solution with sigma large enough to matter."""
    sigma = request.param
    return sigma, optimize(problem["G"], problem["I"], problem["I_std"], sigma=sigma)


def objective_gradient(problem, xi, b, w_list, sigma):
    """
    Central-difference gradient of the regularised objective with respect to
    (xi, b, w_1, ..., w_k), and the objective value. Steps are relative for xi
    and b, which have very different magnitudes from the weights.
    """
    sizes = [w.size for w in w_list]
    x = np.concatenate([[xi, b], *w_list])

    def f(x):
        w = np.split(x[2:], np.cumsum(sizes)[:-1])
        return objective(problem, x[0], x[1], w, sigma)[2]

    grad = np.empty_like(x)
    for i in range(x.size):
        h = 1e-6 * abs(x[i]) if i < 2 else 1e-7
        e = np.zeros_like(x)
        e[i] = h
        grad[i] = (f(x + e) - f(x - e)) / (2 * h)
    return grad, f(x)


class TestRegularisation:
    """
    Test the regularised path (sparse coordinate Jacobian) against the
    unregularised path (dense Jacobian).
    """

    def test_negligible_sigma_reproduces_the_unregularised_solution(self, problem):
        """
        Test that with sigma ~ 0 the coordinate-format Jacobian gives the same
        answer as the dense one, which checks its row/column indexing of the
        data-misfit block
        """
        xi0, b0, w0 = problem["unregularised"]
        xi, b, w_list = optimize(problem["G"], problem["I"], problem["I_std"], sigma=1e-8)

        assert xi == pytest.approx(xi0, rel=1e-6, abs=0)
        assert b == pytest.approx(b0, rel=1e-6)
        for w, w_ref in zip(w_list, w0):
            np.testing.assert_allclose(w, w_ref, atol=1e-6)

    def test_regularisation_trades_fit_for_smoothness(self, problem, regularised):
        """
        Test that the regularised solution is smoother but fits worse than the
        unregularised one, as it must if both are optimal
        """
        sigma, (xi, b, w_list) = regularised
        chi2_reg, rough_reg, _ = objective(problem, xi, b, w_list, sigma)
        chi2_unreg, rough_unreg, _ = objective(problem, *problem["unregularised"], sigma)

        assert rough_reg < rough_unreg
        assert chi2_reg > chi2_unreg

    def test_regularised_solution_beats_the_unregularised_and_true_points(
        self, problem, regularised
    ):
        """
        Test that the regularised objective is lower at the regularised
        solution than at the unregularised solution or the ground truth
        """
        sigma, (xi, b, w_list) = regularised
        _, _, f_reg = objective(problem, xi, b, w_list, sigma)
        _, _, f_unreg = objective(problem, *problem["unregularised"], sigma)
        _, _, f_true = objective(problem, XI_TRUE, B_TRUE, problem["w_true"], sigma)

        assert f_reg < f_unreg
        assert f_reg < f_true

    def test_regularised_solution_satisfies_first_order_optimality(
        self, problem, regularised
    ):
        """
        Test the KKT conditions for min f(xi, b, w) with each w on a simplex,
        using a gradient of the independently computed objective:
        df/dxi = df/db = 0, and within each simplex all weights in use share
        the smallest gradient component (no mass can be moved downhill).
        A wrong regularisation Jacobian would leave the solver short of this.

        The unregularised solution evaluated on the same objective fails the
        simplex condition by about four orders of magnitude more than this
        tolerance.
        """
        sigma, (xi, b, w_list) = regularised
        grad, f = objective_gradient(problem, xi, b, w_list, sigma)
        tol = 1e-5 * (1 + f)

        assert abs(grad[0] * xi) < tol
        assert abs(grad[1] * b) < tol
        sizes = [w.size for w in w_list]
        for w, g in zip(w_list, np.split(grad[2:], np.cumsum(sizes)[:-1])):
            in_use = w > 1e-6 * w.max()
            assert g[in_use].max() - g.min() < tol
