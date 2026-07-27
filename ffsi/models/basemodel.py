"""
SAS Model base class

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
from abc import ABC, abstractmethod

from ffsi.utils import smear_tensor_1d, smear_tensor_2d


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
        pass

    @classmethod
    def compute_smeared_scattering_intensity(cls, q_calc_list, q_calc_weights, param_list, drho):
        """
        Compute q resolution smeared scattering intensity (Green's tensor) `G`

        :param q_calc_list: `list` of resolution smearing scattering vectors `q`
        :param q_calc_weights: weights for q resolution smearing
        :param param_list: `list` of model parameters
        :param drho: difference between scattering length densities
        :return: the resolution smeared scattering intensity (Green's tensor) `G`
        """
        G = cls.compute_scattering_intensity(q_calc_list, param_list, drho)
        if '2D' in cls.__name__:
            return smear_tensor_2d(G, q_calc_weights)
        else: # 1D model
            return smear_tensor_1d(G, q_calc_weights)

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
        pass
