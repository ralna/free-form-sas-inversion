# Free-Form SAS Inversion for SAXS/SANS
[![Tests](https://github.com/ralna/free-form-sas-inversion/actions/workflows/tests.yml/badge.svg)](https://github.com/ralna/free-form-sas-inversion/actions/workflows/tests.yml)

A package for the *free-form* inversion of Small Angle Scattering (SAS) problems arising from X-ray and Neutron sources.

This is a much improved algorithm and implementation over that proposed in the paper:
https://journals.iucr.org/j/issues/2022/04/00/jl5041/index.html

## Installation
```bash
pip install ffsi
```
Computations run on the GPU automatically when [CuPy](https://cupy.dev) is installed, and on the CPU with NumPy otherwise.
To use an NVIDIA GPU, install the CuPy package matching your CUDA version (shown by `nvidia-smi`), e.g. `pip install cupy-cuda13x` for CUDA 13 or `pip install cupy-cuda12x` for CUDA 12.

## Quick Start
Invert 1D SANS or SAXS data for a free-form distribution of sphere radii:
```python
import numpy as np
from ffsi.api import invert

# three columns: q (1/A), I(q) and its standard deviation dI(q)
q, intensity, intensity_std = np.loadtxt("data.txt", unpack=True)

result = invert(
    "sphere", q, intensity, intensity_std,
    grids={"r": (400, 800, 1000)},  # radius bins: (min, max, number of bins) in A
    sld=1.0, sld_solvent=0.0,       # scattering length densities in 1e-6/A^2
    sigma=5.0,                      # smoothing weight (None for no smoothing)
)

r = result.distribution("r")
print(f"volume fraction {result.scale:.3g}, chi2 {result.chi2:.3g}")
print(f"most probable radius {r.grid[np.argmax(r.weights)]:.0f} A")
```
The other models are `"cylinder"` and `"ellipsoid"` (1D), and `"cylinder2d"` and `"ellipsoid2d"` (2D, with `q` given as the `(qx, qy)` axes of the detector grid);
`grids` takes one entry per model parameter (`r` for spheres; `l` and `r` for cylinders; `rp` and `re` for ellipsoids; plus the orientation angles `theta` and `phi`, in radians, for the 2D models). See `help(invert)` for resolution smearing and the full list of options.

## SAS Inversion
Small Angle Scattering is used to probe and analyse the nanoscale structure of materials.
An excellent introduction to (fixed-form) SAS inversion can be found in:
https://journals.iucr.org/j/issues/2021/06/00/gj5274/gj5274.pdf

## Free-Form SAS Inversion
Free-form SAS inversion seeks to determine the distributions of structural properties of nanoparticles (e.g. radius, length),
but crucially (unlike conventional SAS inversion) does not make any assumptions about the forms these distributions take (hence *free-form*).

A Small Angle Scattering experiment records the intensity, i.e. squared amplitude, $I(q)$ of the scattered wave as a function of the scattering vector $q$.
The scattering vector $q$ is itself a function of the small angle $\theta$ and is given by
```math
q = \frac{4 \pi}{\lambda} \sin\left(\frac{\theta}{2}\right)
```
where $\lambda$ is the wavelength of the radiation source. For small angles, $\sin(\theta) \approx \theta$ and the scattering vector reduces to 
```math
q \approx \frac{2 \pi}{\lambda}\theta
```
i.e. essentially the small angle normalised by the wavelength of the radiation. 

### Form Factors (aka Green's Functions)
The form factor $F(q)$ describes the intensity $I(q)$ of scattering from a *single* nanoparticle over a full range of q and constitutes its SAS fingerprint.
For example, the form factor $F(q)$ for a sphere of radius $r$ is given via $j_1(z)$ the spherical Bessel function of the first kind as
```math
F(q) = \left[ V(r)\Delta\rho \dfrac{3j_1(qr)}{qr} \right]^2
```
where $V$ is the sphere volume and $\Delta\rho$ the scattering length density difference.

This repository currently contains the form factors for:

- [Sphere (1D)](https://www.sasview.org/docs/user/models/sphere.html) 
- [Cylinder (1D and 2D)](https://www.sasview.org/docs/user/models/cylinder.html)
- [Ellipsoid (1D and 2D)](https://www.sasview.org/docs/user/models/ellipsoid.html)

More form-factors will be added as the project progresses (as the code in this repository is vectorized and GPU accelerated, we are unable to directly use the form factors from SasView).
They nevertheless reproduce the intensities of SasView's model library, sasmodels (version 1.1, bundled with SasView 6.2), to rounding error, including its adaptive Gauss-Legendre orientation averaging for the 1D cylinder and ellipsoid, and are tested against it.

### Polydispersity
In a real SAS experiment, we are likely to see a population of nanoparticles that possess size and/or orientational distributions, this is called *polydispersity*.
The resultant intensity $I(q)$ is then averaged over the distributions. 
For example, for spheres with different radii,
```math
I(q) = \int F(q,r) w(r) dr
```
where $w(r)$ is the distribution over the radius.

## Running the Tests
```bash
pip install -e . pytest sasmodels
pytest tests
```
The comparisons against SasView's sasmodels are skipped if it is not installed.
