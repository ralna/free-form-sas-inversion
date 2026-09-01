"""
SAS Model base class

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
from abc import ABC, abstractmethod

class SASModel(ABC):

    """
    Names of parameters used for computing the scattering intensity
    (Green's tensor) `G` ordered by their index in `G`
    """
    param_names_scattering_intensity = []

    @staticmethod
    @abstractmethod
    def compute_scattering_intensity(q_list, param_list, drho):
        """
        Compute the scattering intensity (Green's tensor) `G`

        :param q_list: `list` of scattering vectors `q`
        :param param_list: `list` of model parameters
        :param drho: difference between scattering length densities
        :return: the scattering intensity (Green's tensor) `G`
        """
        raise NotImplementedError

    """
    Names of parameters used for computing the average volume `V`
    ordered by their index in `V`
    """
    param_names_average_volume = []

    @staticmethod
    @abstractmethod
    def compute_average_volume(param_list, w_list):
        """
        Compute the volume `V` averaged across parameters

        :param param_list: `list` of model parameters
        :param w_list: `list` of parameter distributions
        :return: the average volume `V`
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def compute_volume(param_list):
        """
        Compute the volume `V` for each combination of model parameters

        :param param_list: `list` of model parameters
        :return: the volume `V`
        """
        raise NotImplementedError
