# Free-Form SAS Inversion for SAXS/SANS
A package for the *free-form* inversion of Small Angle Scattering (SAS) problems arising from X-ray and Neutron sources.

This is a much improved algorithm and implementation over that proposed in the paper:
https://journals.iucr.org/j/issues/2022/04/00/jl5041/index.html

## SAS Inversion
Small Angle Scattering is used to probe and analyse the nanoscale structure of materials.
An excellent introduction to (fixed-form) SAS inversion can be found in:
https://journals.iucr.org/j/issues/2021/06/00/gj5274/gj5274.pdf

## Free-Form SAS Inversion
Free-form SAS inversion seeks to determine the distributions of structural properties of nanoparticles (e.g. radius, length),
but cruically (unlike conventional SAS inversion) does not make any assumptions about the forms these distributions take (hence *free-form*).

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

More form-factors will be added as the project progresses (as the code in this repository is vectorized and GPU accelerated, we are unable to directly use the form factors from SASView).

### Polydispersity
In a real SAS experiment, we are likely to see a population of nanoparticles that possess size and/or orientational distributions, this is called *polydispersity*.
The resultant intensity $I(q)$ is then averaged over the distributions. 
For example, for spheres with different radii,
```math
I(q) = \int F(q,r) w(r) dr
```
where $w(r)$ is the distribution over the radius.
