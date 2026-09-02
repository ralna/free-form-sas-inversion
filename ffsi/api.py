"""
Public API for free-form SAS inversion.
"""
import importlib.util
from dataclasses import dataclass, field
from importlib import import_module
from inspect import getmembers, isabstract, isclass

import numpy as np

from ffsi.array_module import get_array_module, to_device, from_device
from ffsi.models.basemodel import SASModel
from ffsi.optimize_galahad import optimize
from ffsi.utils import contract_tensor, xi_to_scale


# Model names available through invert(), for error messages only
_MODEL_NAMES = ("sphere", "cylinder", "cylinder2d", "ellipsoid", "ellipsoid2d")


def _resolve_model(model):
    """
    Resolve `model` (a case-insensitive name or a `SASModel` subclass) to
    `(class, name)`.

    The model class is imported lazily from its own module `ffsi.models.<name>`
    (module name == class name lowercased), so only the requested model is
    loaded rather than every model up front.
    """
    name = (model if isinstance(model, str) else model.__name__).lower()

    # return a proper error
    if importlib.util.find_spec(f"ffsi.models.{name}") is None:
        raise ValueError(
            "Unknown model '{}', available models: {}".format(
                name, ", ".join(_MODEL_NAMES)
            )
        )

    module = import_module(f"ffsi.models.{name}")

    classes = getmembers(
        module,
        lambda m: (
            isclass(m)
            and not isabstract(m)
            and issubclass(m, SASModel)
            and m is not SASModel
            and m.__module__ == module.__name__
        ),
    )
    if not classes:
        raise ValueError(f"Module 'ffsi.models.{name}' defines no SASModel subclass")

    return classes[0][1], name


@dataclass
class ParamDistribution:
    """Fitted distribution of one model parameter."""

    name: str  # model parameter name, e.g. 'r', 'l', 'rp', 're'
    grid: np.ndarray  # bin centers
    weights: np.ndarray  # weights

    volume_weights: np.ndarray = None


@dataclass
class InversionResult:
    """Output of `invert()`: I_opt = xi * Gw + background."""

    model: str  # model name
    xi: float                # raw scale factor
    background: float
    distributions: list = field(default_factory=list)
    theory: np.ndarray = None  # I_opt on the input q
    residuals: np.ndarray = None  # (theory - intensity) / intensity_std
    chi2: float = None  # sum(residuals**2) / residuals.size
    average_volume: float = None  # <V> under the optimal weights
    drho: float = None  # contrast: sld - sld_solvent
    scale: float = None  # volume fraction: xi * <V> * 1e4

    def distribution(self, name):
        """The fitted `ParamDistribution` for parameter `name`."""
        for dist in self.distributions:
            if dist.name == name:
                return dist
        raise KeyError("No distribution for parameter '{}', have: {}".format(name, ", ".join(d.name for d in self.distributions)))


def _build_grid(spec, xp):
    """Bin centers (on backend `xp`) from a (min, max, nbins) triple"""
    if isinstance(spec, np.ndarray) or (np.ndim(spec) == 1 and len(spec) > 3):
        return xp.asarray(spec, dtype=float)
    lo, hi, nbins = spec
    return xp.linspace(float(lo), float(hi), int(nbins))


def invert(model, q, intensity, intensity_std, grids, *, sld, sld_solvent, sigma=None):
    """
    Free-form inversion of 1D SAS data.

    :param model: model name ('sphere', 'cylinder', 'ellipsoid';
    :param q: scattering vectors
    :param intensity: measured intensity `I(q)`
    :param intensity_std: intensity standard deviations `dI(q)`
    :param grids: `dict` keyed by the model's parameter names; each value
        is a `(min, max, nbins)` triple or a prebuilt 1D array of bin centers
    :param sld: scattering length density of the particle,
        with `sld_solvent` it gives the contrast `drho = sld - sld_solvent`
    :param sld_solvent: scattering length density of the solvent, in 1e-6 A^-2
    :param sigma: smoothness regularization weight (`None` disables it)
    :return: an `InversionResult`; `scale` is the volume fraction `xi * <V> * 1e4`

    Computation runs on the GPU automatically whenever CuPy is installed; inputs
    may be plain numpy arrays (they are moved onto the GPU here), so callers need
    no CuPy dependency of their own. Without CuPy it runs on the CPU.
    """

    # resolve a name or a SASModel subclass to (class, name)
    model_class, model_name = _resolve_model(model)
    param_names = list(model_class.param_names_scattering_intensity)

    # contrast: drho = sld - sld_solvent
    drho = float(sld) - float(sld_solvent)

    # move host inputs onto the compute backend (GPU when CuPy is available)
    # array-module dispatch below and everything downstream run on that backend
    q, intensity, intensity_std = to_device(q, intensity, intensity_std)

    # xp resolves to cupy when the inputs are on the GPU, else numpy
    xp = get_array_module(q, intensity, intensity_std)
    q = xp.ascontiguousarray(q, dtype=float)
    intensity = xp.asarray(intensity, dtype=float)
    intensity_std = xp.asarray(intensity_std, dtype=float)
    # build grids on the same backend
    param_list = [_build_grid(grids[name], xp) for name in param_names]

    # scattering intensity (Green's tensor) and inversion
    G = model_class.compute_scattering_intensity([q], param_list, drho)
    xi, background, w_opt_list = optimize(G, intensity, intensity_std, sigma=sigma)
    xi, background = float(xi), float(background)
    # GALAHAD returns numpy weights; move them onto G's backend to reconstruct
    w_list = [xp.asarray(w) for w in w_opt_list]

    # fitted intensity, residuals and chi-squared
    theory = xi * contract_tensor(G, w_list, skip_axes=[0]) + background
    residuals = (theory - intensity) / intensity_std
    chi2 = float(xp.sum(residuals**2) / residuals.size)

    # average volume
    volume_params = [param_list[param_names.index(name)]
                     for name in model_class.param_names_average_volume]
    volume_weights_list = [w_list[param_names.index(name)]
                           for name in model_class.param_names_average_volume]
    average_volume = float(model_class.compute_average_volume(volume_params, volume_weights_list))

    # convert xi to SasView scale
    scale = xi_to_scale(xi, average_volume)

    # package results as host numpy as plotters and the GUI
    # cannot take cupy arrays
    distributions = []
    for name, grid, weights in zip(param_names, param_list, w_list):
        volume_weights = None
        if len(param_names) == 1:
            volume = model_class.compute_volume(param_list)
            weighted = weights * volume
            volume_weights = from_device(weighted / xp.sum(weighted))
        distributions.append(
            ParamDistribution(
                name=name,
                grid=from_device(grid),
                weights=from_device(weights),
                volume_weights=volume_weights,
            )
        )

    return InversionResult(
        model=model_name,
        xi=xi,
        background=background,
        distributions=distributions,
        theory=from_device(theory),
        residuals=from_device(residuals),
        chi2=chi2,
        average_volume=average_volume,
        drho=drho,
        scale=scale,
    )
