Global Sky Model v3: eGSM 
====================
(under construction)

**Update Note**
* monopole calibration 
* main change on interpolator
* Monte Carlo Prediction

**Description**

eGSM is a Python package for building a foreground sky from 22MHz to 3THz. The algorithm is capable of computing a sky model at any frequency and its error estimate. To support a wide range of frequencies more accurately, we offer two separate models covering different frequency ranges: low (22MHz to 30GHz) and high (22GHz to 3THz).


Installation
------------
eGSM can be installed via pip (in preparation)

`>>> pip install egsm`

To install from the respository, run:

`>>> python setup.py install`

**Package Dependenciees**

Requires:
* numpy >= 1.17
* matplotlib
* astropy >= 1.1
* sklearn
* healpy
