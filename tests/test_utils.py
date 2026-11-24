"""
Tests for the tensor utilities in `ffsi.utils` (contraction, resolution
smearing, xi/scale conversion) against explicit loop implementations.
"""
import numpy as np
import pytest

from ffsi.utils import (
    contract_tensor,
    scale_to_xi,
    smear_tensor_1d,
    smear_tensor_2d,
    xi_to_scale,
)


@pytest.fixture(autouse=True)
def force_cpu(monkeypatch):
    """Run on numpy so results are host arrays."""
    monkeypatch.setattr("ffsi.array_module.CUPY_INSTALLED", False)


@pytest.fixture
def rng():
    return np.random.default_rng(2026)


class TestContractTensor:
    """
    Test that contract_tensor sums G against one weight vector per contracted
    axis and keeps the skipped axes, by comparing with explicit loops.
    """

    def test_contract_tensor_1d_q_matches_loops(self, rng):
        """
        Test the 1D layout G[q, a, b] contracted over a and b
        """
        G = rng.random((5, 3, 4))
        wa, wb = rng.random(3), rng.random(4)

        expected = np.zeros(5)
        for i in range(3):
            for j in range(4):
                expected += G[:, i, j] * wa[i] * wb[j]

        np.testing.assert_allclose(
            contract_tensor(G, [wa, wb], skip_axes=[0]), expected, rtol=1e-14
        )

    def test_contract_tensor_2d_q_matches_loops(self, rng):
        """
        Test the 2D layout G[qx, qy, a, b, c] contracted over a, b and c
        """
        G = rng.random((4, 3, 2, 3, 2))
        wa, wb, wc = rng.random(2), rng.random(3), rng.random(2)

        expected = np.zeros((4, 3))
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    expected += G[:, :, i, j, k] * wa[i] * wb[j] * wc[k]

        np.testing.assert_allclose(
            contract_tensor(G, [wa, wb, wc], skip_axes=[0, 1]), expected, rtol=1e-14
        )

    def test_contract_tensor_keeps_a_skipped_parameter_axis(self, rng):
        """
        Test that skipping a parameter axis (as the solver Jacobian does)
        leaves that axis in place and contracts only the others
        """
        G = rng.random((5, 3, 4))
        wa = rng.random(3)

        expected = np.zeros((5, 4))
        for i in range(3):
            expected += G[:, i, :] * wa[i]

        np.testing.assert_allclose(
            contract_tensor(G, [wa], skip_axes=[0, 2]), expected, rtol=1e-14
        )

    def test_contract_tensor_returns_G_for_an_empty_weight_list(self, rng):
        """
        Test that nothing is contracted when there are no weights
        """
        G = rng.random((5, 3))
        assert contract_tensor(G, [], skip_axes=[0, 1]) is G


class TestSmearTensor1D:
    """
    Test that smear_tensor_1d maps G on the q_calc points to the data q points
    with the (len(q_calc), len(q)) resolution weight matrix.
    """

    def test_smear_tensor_1d_identity_weights_leave_G_unchanged(self, rng):
        """
        Test that identity weights (no smearing) return G
        """
        G = rng.random((6, 3, 2))
        np.testing.assert_allclose(smear_tensor_1d(G, np.eye(6)), G, rtol=1e-15)

    def test_smear_tensor_1d_matches_loops(self, rng):
        """
        Test a general weight matrix, including a change in the number of
        q points (q_calc is usually longer than q)
        """
        G = rng.random((7, 3, 2))
        w = rng.random((7, 4))

        expected = np.zeros((4, 3, 2))
        for j in range(4):
            for i in range(7):
                expected[j] += w[i, j] * G[i]

        np.testing.assert_allclose(smear_tensor_1d(G, w), expected, rtol=1e-14)


class TestSmearTensor2D:
    """
    Test that smear_tensor_2d averages each (qx, qy) point over the nbins x
    nbins block of q_calc points with normalised Gaussian weights.
    """

    def test_smear_tensor_2d_single_bin_leaves_G_unchanged(self, rng):
        """
        Test that one bin (no smearing) returns G
        """
        G = rng.random((4, 3, 2))
        np.testing.assert_allclose(smear_tensor_2d(G, np.array([1.0])), G, rtol=1e-15)

    def test_smear_tensor_2d_delta_weights_pick_one_block(self, rng):
        """
        Test that a delta weight vector selects the matching block of q_calc
        """
        nbins, nqx, nqy = 3, 4, 2
        G = rng.random((nbins * nqx, nbins * nqy, 5))

        smeared = smear_tensor_2d(G, np.array([0.0, 1.0, 0.0]))

        np.testing.assert_allclose(
            smeared, G[nqx:2 * nqx, nqy:2 * nqy], rtol=1e-15
        )

    def test_smear_tensor_2d_matches_loops(self, rng):
        """
        Test general weights against explicit loops, including the division by
        sum(w)^2 that makes the weights unnormalised on input
        """
        nbins, nqx, nqy = 3, 4, 2
        G = rng.random((nbins * nqx, nbins * nqy, 5))
        w = rng.random(nbins)

        expected = np.zeros((nqx, nqy, 5))
        for i in range(nbins):
            for k in range(nbins):
                block = G[i * nqx:(i + 1) * nqx, k * nqy:(k + 1) * nqy]
                expected += w[i] * w[k] * block
        expected /= w.sum() ** 2

        np.testing.assert_allclose(smear_tensor_2d(G, w), expected, rtol=1e-14)


class TestScaleConversion:
    """
    Test the conversion between ffsi's xi and SasView's volume-fraction scale.
    """

    def test_scale_to_xi_known_value(self):
        """
        Test xi = 1e-4 * scale / V
        """
        assert scale_to_xi(0.02, 4.0e5) == pytest.approx(5.0e-12, rel=1e-15)

    @pytest.mark.parametrize("scale", [1e-6, 0.01, 1.0])
    def test_scale_and_xi_round_trip(self, scale):
        """
        Test that the two conversions invert each other
        """
        volume = 5.24e8
        assert xi_to_scale(scale_to_xi(scale, volume), volume) == pytest.approx(
            scale, rel=1e-14
        )
