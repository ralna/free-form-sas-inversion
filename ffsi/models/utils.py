"""
SAS Model utility functions

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
from ffsi.models import gauss_legendre_tables as tables

def gauss_legendre(xp, qr_max, elements_per_point=0):
    r"""
    Adaptive Gauss-Legendre quadrature points and weights on [-1,1]

    The number of points is chosen for each :math:`q r_{max}` (the scattering
    vector times the largest particle dimension) as in SasView (sasmodels
    >= 1.1.0, sasmodels/models/lib/adaptive.c), since the orientation
    integrand oscillates faster as :math:`q r_{max}` grows:

    ==========================  ======
    :math:`q r_{max}`           points
    ==========================  ======
    :math:`[0, 10)`             20
    :math:`[10, 100)`           76
    :math:`[100, 750)`          500
    :math:`[750, \infty)`       5000
    ==========================  ======

    An n point rule is symmetrically distributed around 0 and
    can exactly integrate polynomials up to degree 2n-1

    The Gaussian quadrature rule on [a,b] is then:

    .. math::

        \int_a^b f(x) dx \approx \frac{b-a}{2} \sum_i w_i f(\frac{b-a}{2} z_i + \frac{a+b}{2})

    Each rule is returned in blocks of points, so that evaluating a block at
    once vectorises over the points without exceeding 2^24 array elements
    (128 MB of float64) per intermediate array. Summing over all the blocks
    for an index gives the full quadrature rule for those entries.

    :param xp: array module (numpy or cupy)
    :param qr_max: array of :math:`q r_{max}` values
    :param elements_per_point: elements per quadrature point in the caller's
                               largest array when this exceeds the number of
                               selected entries (sets the block size)
    :return: generator of `(index, z, w)` where `index` (as from `nonzero`)
             selects the entries of `qr_max` using the rule and `z` and `w`
             are a block of its points and weights
    """

    # number of quadrature points for each entry of qr_max
    n_points = xp.where(qr_max < 10, 20,
                        xp.where(qr_max < 100, 76,
                                 xp.where(qr_max < 750, 500, 5000)))

    for n in (20, 76, 500, 5000):

        # entries of qr_max using the n point rule
        index = xp.nonzero(n_points == n)
        if index[0].size == 0:
            continue

        # n point quadrature points and weights
        z, w = _gauss_legendre_rule(xp, n)

        # split the points into blocks of at most 2^24 array elements
        size = max(1, 2**24 // max(index[0].size, elements_per_point))
        for start in range(0, n, size):
            block = slice(start, start + size)
            yield index, z[block], w[block]

def _gauss_legendre_rule(xp, n):
    """
    n point Gauss-Legendre quadrature points and weights on [-1,1]
    from the SasView tables (see `ffsi.models.gauss_legendre_tables`)
    """

    # 20 point rule is stored in full
    if n == 20:
        return xp.array(tables.GAUSS20_Z), xp.array(tables.GAUSS20_W)

    # other rules are stored for one side
    z_side = xp.array(getattr(tables, f'GAUSS{n}_Z_SIDE'))
    w_side = xp.array(getattr(tables, f'GAUSS{n}_W_SIDE'))

    # quadrature points (both sides)
    z = xp.concatenate((z_side, xp.flip(-z_side)))

    # quadrature weights (both sides)
    w = xp.concatenate((w_side, xp.flip(w_side)))

    return z, w
