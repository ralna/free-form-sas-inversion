"""
Tests for `ffsi.sensitivity_analysis.sensitivity`: the parameter sensitivity
H (x / g), where g and H are the gradient and Hessian of 0.5 ||r||^2, checked
against finite differences of an independently computed residual; and the
posterior standard deviations, checked against their closed forms.
"""
import numpy as np
import pytest

from ffsi.models.cylinder import Cylinder
from ffsi.models.cylinder2d import Cylinder2D
from ffsi.models.sphere import Sphere
from ffsi.sensitivity_analysis import sensitivity
from ffsi.utils import contract_tensor


@pytest.fixture(autouse=True)
def force_cpu(monkeypatch):
    """Run on numpy so results are host arrays."""
    monkeypatch.setattr("ffsi.array_module.CUPY_INSTALLED", False)


def _sphere():
    q = np.logspace(-3, -0.5, 40)
    return [q], Sphere.compute_scattering_intensity([q], [np.linspace(20, 80, 6)], 1.0)


def _cylinder():
    q = np.logspace(-3, -0.5, 40)
    grids = [np.linspace(200, 600, 5), np.linspace(5, 35, 4)]
    return [q], Cylinder.compute_scattering_intensity([q], grids, 1.0)


def _cylinder2d():
    q = np.linspace(-0.1, 0.1, 6)
    grids = [
        np.linspace(100, 140, 3),
        np.linspace(20, 30, 2),
        np.deg2rad(np.linspace(20, 80, 3)),
        np.deg2rad(np.linspace(0, 180, 4)),
    ]
    return [q, q], Cylinder2D.compute_scattering_intensity([q, q], grids, 1.0)


# 1, 2 and 4 weight vectors; the 2D case also has 2 q axes
SETUPS = {"sphere": _sphere, "cylinder": _cylinder, "cylinder2d": _cylinder2d}


@pytest.fixture(scope="module", params=SETUPS)
def case(request):
    """
    G, synthetic data and a deliberately non-optimal point (xi, b, w), so
    that every gradient component is nonzero and x / g is well defined.
    """
    q_list, G = SETUPS[request.param]()
    rng = np.random.default_rng(7)
    n_q = len(q_list)
    p_dims = G.shape[n_q:]

    w_true = [rng.dirichlet(np.ones(n)) for n in p_dims]
    gw = contract_tensor(G, w_true, skip_axes=list(range(n_q)))
    xi_true = 1.0 / gw.mean()
    intensity = xi_true * gw + 0.1
    intensity_std = (0.05 + 0.05 * rng.random(intensity.shape)) * intensity

    w = [rng.dirichlet(np.ones(n)) for n in p_dims]
    return {
        "G": G,
        "I": intensity,
        "I_std": intensity_std,
        "xi": 1.3 * xi_true,
        "b": 0.07,
        "w": w,
        "q_axes": list(range(n_q)),
    }


def half_sum_of_squares(case, x):
    """0.5 ||r||^2 at the flattened point x = (xi, b, w_1, ..., w_k)."""
    sizes = [w.size for w in case["w"]]
    w_list = np.split(x[2:], np.cumsum(sizes)[:-1])
    model = x[0] * contract_tensor(case["G"], w_list, skip_axes=case["q_axes"]) + x[1]
    r = (model - case["I"]) / case["I_std"]
    return 0.5 * np.sum(r**2)


def finite_difference_derivatives(case, x):
    """
    Central-difference gradient and Hessian of 0.5 ||r||^2.

    r is linear in each variable separately, so 0.5 ||r||^2 is biquadratic in
    any pair of variables and these central differences have no truncation
    error. The steps can therefore be large (10% of each variable, as xi is
    many orders smaller than b and the weights), which keeps rounding error
    small even after the cancellation in H (x / g).
    """
    n = x.size
    h = 0.1 * np.abs(x)
    f = lambda y: half_sum_of_squares(case, y)  # noqa: E731

    grad = np.empty(n)
    hess = np.empty((n, n))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h[i]
        grad[i] = (f(x + ei) - f(x - ei)) / (2 * h[i])
        for j in range(i, n):
            ej = np.zeros(n)
            ej[j] = h[j]
            hess[i, j] = hess[j, i] = (
                f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)
            ) / (4 * h[i] * h[j])
    return grad, hess


class TestSensitivity:
    """
    Test the sensitivity H (x / g) returned by sensitivity() for one, two and
    four weight vectors (the latter on 2D data), which exercises the Jacobian
    and every block of the second-order Hessian term.
    """

    def test_sensitivity_matches_finite_differences(self, case):
        xi, b, w = case["xi"], case["b"], case["w"]
        x = np.concatenate([[xi, b], *w])
        grad, hess = finite_difference_derivatives(case, x)
        expected = hess @ (x / grad)

        sens_xi, sens_b, sens_w = sensitivity(
            case["G"], case["I"], case["I_std"], xi, b, w, compute_uncertainty=False
        )
        actual = np.concatenate([[sens_xi, sens_b], *sens_w])

        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_sensitivity_without_uncertainty_returns_three_values(self, case):
        result = sensitivity(
            case["G"], case["I"], case["I_std"], case["xi"], case["b"], case["w"],
            compute_uncertainty=False,
        )

        assert len(result) == 3
        assert [s.shape for s in result[2]] == [w.shape for w in case["w"]]


@pytest.fixture(scope="module")
def uncertainty(case):
    """The posterior standard deviations (std_xi, std_b, std_w) at the case's point."""
    _, _, _, std_xi, std_b, std_w = sensitivity(
        case["G"], case["I"], case["I_std"], case["xi"], case["b"], case["w"]
    )
    return std_xi, std_b, std_w


class TestUncertainty:
    """
    Test the posterior standard deviations returned by sensitivity() against
    their closed forms (Tarantola eq. 3.56, with a sparse approximate inverse
    for the weights).
    """

    def test_std_of_xi_and_b_match_closed_forms(self, case, uncertainty):
        """
        Test std(xi) = 1 / ||Gw / sigma|| and std(b) = 1 / ||1 / sigma||
        """
        std_xi, std_b, _ = uncertainty
        gw = contract_tensor(case["G"], case["w"], skip_axes=case["q_axes"])

        assert std_xi == pytest.approx(
            1 / np.sqrt(np.sum((gw / case["I_std"]) ** 2)), rel=1e-12, abs=0
        )
        assert std_b == pytest.approx(
            1 / np.sqrt(np.sum(case["I_std"] ** -2.0)), rel=1e-12
        )

    def test_std_of_weights_match_closed_form(self, case, uncertainty):
        """
        Test std(w_k) = sqrt(diag(A) / colsum(A * A)), A = J^T C^-1 J, where
        column j of J is xi * dGw/dw_k[j], built here with explicit slicing
        """
        _, _, std_w = uncertainty
        n_q = len(case["q_axes"])
        c_inv = case["I_std"].reshape(-1) ** -2.0

        assert len(std_w) == len(case["w"])
        for k, w_k in enumerate(case["w"]):
            others = [w for i, w in enumerate(case["w"]) if i != k]
            columns = []
            for j in range(w_k.size):
                G_j = np.take(case["G"], j, axis=n_q + k)
                columns.append(
                    case["xi"] * contract_tensor(G_j, others, skip_axes=case["q_axes"])
                )
            J = np.stack([c.reshape(-1) for c in columns], axis=1)
            A = J.T @ (c_inv[:, None] * J)
            expected = np.sqrt(np.diag(A) / (A * A).sum(axis=0))

            np.testing.assert_allclose(std_w[k], expected, rtol=1e-10)
            assert (std_w[k] > 0).all()
