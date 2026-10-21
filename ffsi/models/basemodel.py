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
        raise NotImplementedError

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

    @classmethod
    def compute_binned_smeared_scattering_intensity(
        cls, q_calc_list, bin_weights, param_list, drho
    ):
        """
        Compute the q resolution smeared scattering intensity (Green's tensor)
        `G` for per-point resolution sampling, as produced by sasmodels
        `Pinhole2D` (`q_calc`, `q_calc_weights`).

        :param q_calc_list: `list` of sample point coordinates, each of length
            `nbins * nq`: `[q]` for 1D models (e.g. `|q|` of 2D data for an
            isotropic model) or the paired `[qx, qy]` for 2D models
        :param bin_weights: 1D weights of the `nbins` resolution bins
        :param param_list: `list` of model parameters
        :param drho: difference between scattering length densities
        :return: the smeared Green's tensor, shape `(nq, ...)`
        """
        nbins = len(bin_weights)
        n_calc = len(q_calc_list[0])
        if nbins == 0 or n_calc % nbins:
            raise ValueError(
                "q_calc length {} is not a multiple of the number of resolution bins {}".format(
                    n_calc, nbins
                )
            )
        nq = n_calc // nbins
        # 2D models evaluate on paired detector points, not an outer-product grid
        kwargs = {"paired": True} if "2D" in cls.__name__ else {}

        G = None
        for b in range(nbins):
            q_bin = [q[b * nq : (b + 1) * nq] for q in q_calc_list]
            G_bin = cls.compute_scattering_intensity(q_bin, param_list, drho, **kwargs)
            G_bin *= bin_weights[b]
            if G is None:
                G = G_bin
            else:
                G += G_bin
        G /= sum(float(w) for w in bin_weights)
        return G

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
