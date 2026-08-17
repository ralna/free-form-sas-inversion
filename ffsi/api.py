"""
Public API for free-form SAS inversion.
"""
from dataclasses import dataclass, field

import numpy as np

from ffsi.models import get_model
from ffsi.models.basemodel import SASModel
from ffsi.optimize_galahad import optimize
from ffsi.utils import contract_tensor


@dataclass
class ParamDistribution:
    """Fitted distribution of one model parameter."""
    name: str                # model parameter name, e.g. 'r', 'l', 'rp', 're'
    grid: np.ndarray         # bin centers
    weights: np.ndarray      # optimal weights on the simplex (sum to 1)

    volume_weights: np.ndarray = None


@dataclass
class InversionResult:
    """Output of `invert()`: I_opt = xi * Gw + background."""
    model: str               # model name ('' if a class was passed directly)
    xi: float                # raw scale factor
    background: float
    distributions: list = field(default_factory=list)
    theory: np.ndarray = None       # I_opt on the input q
    residuals: np.ndarray = None    # (theory - intensity) / intensity_std
    chi2: float = None              # sum(residuals**2) / residuals.size
    average_volume: float = None    # <V> under the optimal weights
    drho: float = None              # contrast G built with (1.0 if none given)
    scale: float = None

    def distribution(self, name):
        """The fitted `ParamDistribution` for parameter `name`."""
        for dist in self.distributions:
            if dist.name == name:
                return dist
        raise KeyError("No distribution for parameter '{}', have: {}".format(name, ", ".join(d.name for d in self.distributions)))


def _build_grid(spec):
    """Bin centers from a (min, max, nbins) triple or a prebuilt 1D array."""
    if isinstance(spec, np.ndarray) or (np.ndim(spec) == 1 and len(spec) > 3):
        return np.asarray(spec, dtype=float)
    lo, hi, nbins = spec
    return np.linspace(float(lo), float(hi), int(nbins))


def invert(model, q, intensity, intensity_std, grids, *,
           sld=None, sld_solvent=None, sigma=None):
    """
    Free-form inversion of 1D SAS data. Assumes valid inputs.

    :param model: model name (see `ffsi.models.MODEL_REGISTRY`) or a
        `SASModel` subclass
    :param q: scattering vectors
    :param intensity: measured intensity `I(q)`
    :param intensity_std: intensity standard deviations `dI(q)`
    :param grids: `dict` keyed by the model's parameter names; each value
        is a `(min, max, nbins)` triple or a prebuilt 1D array of bin centers
    :param sld: scattering length density of the particle, in 1e-6 A^-2;
        with `sld_solvent` it gives the contrast `drho = sld - sld_solvent`
    :param sld_solvent: scattering length density of the solvent, in 1e-6 A^-2
    :param sigma: smoothness regularization weight (`None` disables it)
    :return: an `InversionResult`; `scale` is a volume fraction if a contrast
        was supplied and `None` otherwise
    """

    # if it's a SASModel subclass, use it directly
    # else, lookup via get_model()
    # if isinstance(model, type) and issubclass(model, SASModel):
    #     model_class, model_name = model, getattr(model, '__name__', '').lower()
    # else:
    model_class, model_name = get_model(model), model
    param_names = list(model_class.param_names_scattering_intensity)

    # contrast: drho = sld - sld_solvent
    contrast_supplied = sld is not None and sld_solvent is not None
    drho = float(sld) - float(sld_solvent) if contrast_supplied else 1.0

    q = np.ascontiguousarray(q, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    intensity_std = np.asarray(intensity_std, dtype=float)
    param_list = [_build_grid(grids[name]) for name in param_names]

    # scattering intensity (Green's tensor) and inversion
    G = model_class.compute_scattering_intensity([q], param_list, drho)
    xi, background, w_list = optimize(G, intensity, intensity_std, sigma=sigma)
    xi, background = float(xi), float(background)
    w_list = [np.asarray(w) for w in w_list]

    # fitted intensity, residuals and chi-squared
    theory = xi * np.asarray(contract_tensor(G, w_list, skip_axes=[0])) + background
    residuals = (theory - intensity) / intensity_std
    chi2 = float(np.sum(residuals ** 2) / residuals.size)

    # average volume
    volume_params = [param_list[param_names.index(name)]
                     for name in model_class.param_names_average_volume]
    volume_weights_list = [w_list[param_names.index(name)]
                           for name in model_class.param_names_average_volume]
    average_volume = float(model_class.compute_average_volume(volume_params, volume_weights_list))

    scale = xi * average_volume * 1e4 if contrast_supplied else None

    distributions = []
    for name, grid, weights in zip(param_names, param_list, w_list):
        volume_weights = None
        if len(param_names) == 1:
            volume = np.asarray(model_class.compute_volume(param_list))
            weighted = weights * volume
            volume_weights = weighted / np.sum(weighted)
        distributions.append(ParamDistribution(name=name, grid=grid, weights=weights,
                                               volume_weights=volume_weights))

    return InversionResult(model=model_name, xi=xi, background=background,
                           distributions=distributions, theory=theory,
                           residuals=residuals, chi2=chi2,
                           average_volume=average_volume,
                           drho=drho, scale=scale)
