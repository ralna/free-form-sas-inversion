"""
SAS Models

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""

# Import models into module namespace
from ffsi.models.sphere import Sphere
from ffsi.models.cylinder import Cylinder
from ffsi.models.cylinder2d import Cylinder2D
from ffsi.models.ellipsoid import Ellipsoid
from ffsi.models.ellipsoid2d import Ellipsoid2D

# Models usable through ffsi.api (1D only for now)
MODEL_REGISTRY = {
    "sphere": Sphere,
    "cylinder": Cylinder,
    "ellipsoid": Ellipsoid,
}


def get_model(name):
    """
        Look up a model class by name
        :return: the `SASModel` subclass
    """
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(
            "Unknown model '%s', available models: %s"
            % (name, ", ".join(sorted(MODEL_REGISTRY)))
        ) from None
