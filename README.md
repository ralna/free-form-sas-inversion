# Free-form SAS Inversion for SAXS/SANS
A package for the free-form inversion of Small Angle Scattering problems arising from X-ray and Neutron sources.

This is an orders of magnitude faster and vastly improved algorithm over that in:
https://journals.iucr.org/j/issues/2022/04/00/jl5041/index.html

## Overview of SAS Inversion
An excellent introduction to (fixed-form) SAS inversion can be found in:
https://journals.iucr.org/j/issues/2021/06/00/gj5274/gj5274.pdf

## Free-form SAS Inversion
The free-form SAS inversion problem does not make any assumptions on the forms of the distributions (hence free-form).

The scattering vector $q$ is a function of the small angle $\theta$ and is given by
```math
q = \frac{4 \pi}{\lambda} \sin(\theta/2)
```
where $\lambda$ is the wavelength of the radiation source. For small angles, $\sin(\theta/2) = \theta/2$ and the scattering vector reduces to
```math
q = \frac{2 \pi \theta}{\lambda} 
```

### Form Factors (aka Green's Functions)
The form factor $P(q)$ describes how radiation scatters from a single particle, revealing its size, shape and internal structure (e.g. a sphere or cylinder) by analyzing intensity patterns at small angles, essentially representing the intensity of scattering from a single particle
```math
I(q) = p^2 V^2 P(q)
```
This repository currently contains the form factors for:

- [Sphere](https://www.sasview.org/docs/user/models/sphere.html) 
- [Cylinder](https://www.sasview.org/docs/user/models/cylinder.html)
- [Ellipsoid](https://www.sasview.org/docs/user/models/ellipsoid.html)

The intention is that these will be replaced by the form factors provided by SASView in due course.

### Structure Factors
The structure factor $S(q)$ describes inter-particle interactions in a sample, complementing the single-particle form factor $P(q)$ to determine the total scattering intensity
```math
I(q) = p^2 V^2 P(q) S(q)
```
revealing particle arrangements beyond just size and shape, with $S(q)=1$ for non-interacting particles.

This repository currently contains the structure factors for:

- [Hard Sphere](https://www.sasview.org/docs/user/models/hardsphere.html)

The intention is that these will be replaced by the structure factors provided by SASView in due course.

## Tensor Trains
We construct a low-rank representation of the form factor using the tensor-train decomposition, for more details please see:
https://tensornetwork.org/mps/
