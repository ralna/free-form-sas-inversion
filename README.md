# Free-form SAS Inversion for SAXS/SANS
A package for the free-form inversion of Small Angle Scattering problems arising from X-ray and Neutron sources.

This is an improved algorithm and implementation over that proposed in:
https://journals.iucr.org/j/issues/2022/04/00/jl5041/index.html

## Overview of SAS Inversion
An excellent introduction to (fixed-form) SAS inversion can be found in:
https://journals.iucr.org/j/issues/2021/06/00/gj5274/gj5274.pdf

## Free-form SAS Inversion
Free-form SAS inversion does not make any assumptions on the forms of the distributions (hence free-form). It seeks to determine the distributions of particle properties in the sample.

A Small Angle Scattering experiment records the intensity, i.e. squared amplitude, $I(q)$ of the scattered wave as a function of the scattering vector $q$.
The scattering vector $q$ is itself a function of the small angle $\theta$ and is given by
```math
q = \frac{4 \pi}{\lambda} \sin\left(\frac{\theta}{2}\right)
```
where $\lambda$ is the wavelength of the radiation source. For small angles, $\sin(\theta) = \theta$ and the scattering vector reduces to $q = \frac{2 \pi}{\lambda}\theta$. 

### Form Factors (aka Green's Functions)
The form factor $P(q)$ describes how radiation scatters from a single particle, revealing its size, shape and internal structure (e.g. a sphere or cylinder) by analysing intensity patterns $I(q)$ at small angles, essentially representing the intensity of scattering from a single particle via
```math
I(q) = \Delta\rho^2 V^2 P(q)
```
where $V$ is the particle volume and $\Delta\rho$ the scattering density difference or contrast. This repository currently contains the form factors for:

- [Sphere](https://www.sasview.org/docs/user/models/sphere.html) 
- [Cylinder](https://www.sasview.org/docs/user/models/cylinder.html)
- [Ellipsoid](https://www.sasview.org/docs/user/models/ellipsoid.html)
