"""
SAS plotting functions

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

### 1D intensity plotting

def plot_1d_intensity(q, I_data, I_data_std=None):
    print('INFO: plotting 1D intensity...')
    plt.figure()
    plt.errorbar(q, I_data, yerr=I_data_std, ecolor='gray')
    plt.grid()
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Intensity')
    plt.xlabel(r"Scattering vector $q$ ($\AA^{-1}$)")
    plt.ylabel(r"Intensity $I$ ($\mathrm{cm}^{-1}$)")
    plt.show()

def plot_1d_optimized_intensity(q, I_data, I_opt, I_data_std=None):
    print('INFO: plotting 1D optimized intensity...')
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

### 1D sphere plotting

def plot_sphere_distribution(r, w_r, title=None, normalize_by_volume=False):
    print('INFO: plotting sphere distribution...')
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

### 2D intensity plotting

def plot_2d_intensity(qx, qy, I_data):
    print('INFO: plotting 2D intensity...')
    plt.figure()
    plt.imshow(I_data.T,
            extent=(qx[0], qx[-1], qy[0], qy[-1]), aspect=1., cmap='turbo',
            norm=colors.LogNorm(vmin=I_data.min(), vmax=I_data.max()))
    plt.xlabel(r"Scattering vector $qx$ ($\AA^{-1}$)")
    plt.ylabel(r"Scattering vector $qy$ ($\AA^{-1}$)")
    plt.title(r"Intensity image $I(q_x, q_y)$ ($\mathrm{cm}^{-1})$")
    plt.colorbar()
    plt.show()

def plot_2d_optimized_intensity(qx, qy, I_data, I_opt):
    print('INFO: plotting 2D optimized intensity...')
    plt.figure()
    plt.imshow(I_opt.T,
            extent=(qx[0], qx[-1], qy[0], qy[-1]), aspect=1., cmap='turbo',
            norm=colors.LogNorm(vmin=I_data.min(), vmax=I_data.max()))
    plt.xlabel(r"Scattering vector $qx$ ($\AA^{-1}$)")
    plt.ylabel(r"Scattering vector $qy$ ($\AA^{-1}$)")
    plt.title(r"Optimized Intensity image $I(q_x, q_y)$ ($\mathrm{cm}^{-1})$")
    plt.colorbar()
    plt.show()

def plot_2d_intensity_misfit(qx, qy, I_data, I_opt):
    print('INFO: plotting 2D intensity misfit...')
    plt.figure()
    plt.imshow(np.abs(I_data.T - I_opt.T),
            extent=(qx[0], qx[-1], qy[0], qy[-1]), aspect=1., cmap='turbo',
            norm=colors.LogNorm(vmin=I_data.min(), vmax=I_data.max()))
    plt.xlabel(r"Scattering vector $qx$ ($\AA^{-1}$)")
    plt.ylabel(r"Scattering vector $qy$ ($\AA^{-1}$)")
    plt.title(r"Intensity Misfit ($\mathrm{cm}^{-1})$")
    plt.colorbar()
    plt.show()

### 2D cylinder plotting

def plot_cylinder_distributions(param_list, w_list, title=None):
    print('INFO: plotting cylinder distributions...')
    fig, ax = plt.subplots(2, 2)
    plt.suptitle(title)
    plt.subplots_adjust(hspace=.5, wspace=.5)
    ax[0,0].plot(param_list[0], w_list[0] * 100) # x100 to percent
    ax[0,1].plot(param_list[1], w_list[1] * 100) # x100 to percent
    ax[1,0].plot(param_list[2], w_list[2] * 100) # x100 to percent
    ax[1,1].plot(param_list[3], w_list[3] * 100) # x100 to percent
    ax[0,0].set_xlabel(r"Length $l$ ($\AA$)")
    ax[0,1].set_xlabel(r"Radius $r$ ($\AA$)")
    ax[1,0].set_xlabel(r"Cylinder axis to beam angle $\theta$ (radians)")
    ax[1,1].set_xlabel(r"Rotation about beam $\phi$ (radians)")
    ax[0,0].set_ylabel(r"Weights $w$ (%)")
    ax[0,1].set_ylabel(r"Weights $w$ (%)")
    ax[1,0].set_ylabel(r"Weights $w$ (%)")
    ax[1,1].set_ylabel(r"Weights $w$ (%)")
    ax[0,0].grid()
    ax[0,1].grid()
    ax[1,0].grid()
    ax[1,1].grid()
    plt.show()

### 2D ellipsoid plotting

def plot_ellipsoid_distributions(param_list, w_list, title=None):
    print('INFO: plotting ellipsoid distributions...')
    fig, ax = plt.subplots(2, 2)
    plt.suptitle(title)
    plt.subplots_adjust(hspace=.5, wspace=.5)
    ax[0,0].plot(param_list[0], w_list[0] * 100) # x100 to percent
    ax[0,1].plot(param_list[1], w_list[1] * 100) # x100 to percent
    ax[1,0].plot(param_list[2], w_list[2] * 100) # x100 to percent
    ax[1,1].plot(param_list[3], w_list[3] * 100) # x100 to percent
    ax[0,0].set_xlabel(r"Polar radius $r_p$ ($\AA$)")
    ax[0,1].set_xlabel(r"Equatorial radius $r_e$ ($\AA$)")
    ax[1,0].set_xlabel(r"Ellipsoid axis to beam angle $\theta$ (radians)")
    ax[1,1].set_xlabel(r"Rotation about beam $\phi$ (radians)")
    ax[0,0].set_ylabel(r"Weights $w$ (%)")
    ax[0,1].set_ylabel(r"Weights $w$ (%)")
    ax[1,0].set_ylabel(r"Weights $w$ (%)")
    ax[1,1].set_ylabel(r"Weights $w$ (%)")
    ax[0,0].grid()
    ax[0,1].grid()
    ax[1,0].grid()
    ax[1,1].grid()
    plt.show()
