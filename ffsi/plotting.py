"""
SAS plotting functions

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import matplotlib.pyplot as plt

def plot_1d_intensities(q, I_data, I_data_std=None):
    plt.figure()
    plt.errorbar(q, I_data, yerr=I_data_std, ecolor='gray')
    plt.grid()
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Intensity')
    plt.xlabel(r"Scattering vector $q$ ($\AA^{-1}$)")
    plt.ylabel(r"Intensity $I$ ($\mathrm{cm}^{-1}$)")
    plt.show()

def plot_optimized_intensities(q, I_data, I_opt, I_data_std=None):
    plt.figure()
    plt.grid()
    plt.errorbar(q, I_data, yerr=I_data_std, ecolor='gray', marker='o', markerfacecolor='none')
    plt.plot(q, I_opt, color='red', zorder=5)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Optimized Intensity')
    plt.xlabel(r"Scattering vector $q$ ($\AA^{-1}$)")
    plt.ylabel(r"Intensity $I$ ($\mathrm{cm}^{-1}$)")
    plt.legend(['Fit','Data'])
    plt.show()

def plot_sphere_distribution(r, w_r, title=None, normalize_by_volume=False):
    plt.figure()
    if normalize_by_volume:
        v = r ** 3 # sphere volume
        w_r_hat = w_r * v / (w_r * v).sum() * 100  # x100 to percent
        cmap = plt.get_cmap('turbo_r') # colormap
        plt.plot(r, w_r_hat, c=cmap(0.0))
    else:
        plt.plot(r, w_r * 100) # x100 to percent
    plt.grid()
    plt.title(title)
    plt.xlabel(r"Radius $r$ ($\AA$)")
    plt.ylabel(r"Weights $w$ (%)")
    plt.show()
