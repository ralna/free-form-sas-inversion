"""
SAS Hard Sphere Structure Factor (1D)
https://www.sasview.org/docs/user/models/hardsphere.html
(adapted from hardsphere.c)

Parameters:
q - scattering vector
r_eff - effective radius of hard sphere
vol_frac - volume fraction of hard sphere

Copyright (C) 2025 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import numpy as np

def S_hard_sphere(q, r_eff, vol_frac):

    if(np.abs(r_eff) < 1e-12):
        return 1.0

    D = (1 / (1 - vol_frac)) ** 2
    A = ((1 + 2 * vol_frac) * D) ** 2
    X = np.abs(q * r_eff * 2)

    if(X < 5e-6):
        return 1/A

    X2 = X * X
    B = -6 * vol_frac * ((1 + 0.5 * vol_frac) * D) ** 2
    G = 0.5 * vol_frac * A

    # use Taylor series expansion for small X
    if (X < 0.05):
        FF = 8 * A + 6 * B + 4 * G + (
             -0.8 * A - B / 1.5 - 0.5 * G +
             (A / 35. + 0.0125 * B + 0.02 * G ) * X2) * X2
        SF = 1 / (1 + vol_frac * FF)
        return SF

    X4 = X2 * X2
    S = np.sin(X)
    C = np.cos(X)

    FF = ((G * ((4 * X2 - 24) * X * S - (
           X4 - 12 * X2 + 24) * C + 24) / X2 + B * (
           2 * X * S - (X2 - 2) * C - 2)) / X + A * (S - X * C)) / X
    SF = 1 / (1 + 24 * vol_frac * FF / X2)

    return SF
