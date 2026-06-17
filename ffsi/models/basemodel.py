"""
SAS Model base class

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
from abc import ABC, abstractmethod

class SASModel(ABC):

    @classmethod
    @abstractmethod
    def compute_G(self, q_list, param_dict, const_dict):
        """
        Compute the Green's tensor `G`

        :param q_list: `list` of scattering vectors `q`
        :param param_dict: `dict` of model parameters
        :param const_dict: `dict` of model constants
        :return: the Green's tensor `G`
        """
        pass

    @classmethod
    @abstractmethod
    def compute_average_V(self, param_dict, w_dict):
        """
        Compute the volume `V` averaged across parameters

        :param param_dict: `dict` of model parameters
        :param w_dict: `dict` of parameter distributions
        :return: the average volume
        """
        pass

    @classmethod
    @abstractmethod
    def get_param_keys_G(self):
        """
        Get the keys of the parameters used for computing `G`

        :return: `list` of parameter keys ordered by their index in `G`
        """
        pass

    @classmethod
    @abstractmethod
    def get_param_keys_V(self):
        """
        Get the keys of the parameters used for computing `V`

        :return: `list` of parameter keys ordered by their index in `V`
        """
        pass
