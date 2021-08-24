# data-xray (a Pythonic cure for hyperspectral morass)

Data-Xray provides a Pythonic interface to organization and analysis 
of data, acquired primarily with Nanonis controllers by Specs GmbH.
The primary emphasis at the moment is on scanning probe microscopy measurements, 
but most of the analysis applies to other domains of hyperspectral data.
Data organization is achieved through the use of xarray  - N-dimensional 
extension of Pandas phenomenology onto arbitrary dimensions.

Presently implemented functionality includes:

(1) import of Nanonis file formats, such as .sxm (images), .dat
(spectroscopy), .3ds (binary spectroscopy files) into python and
conversion of data to xarrays

(2) rapid reporting functionality, with automatic traversal of 
the file system and data summarization in Powerpoint files

(3) data export into netcdf format for interfaces with other data platforms
(Julia, Wolfram Language)

(4) functions for spectral and image visualization, and 
    elements of unsupervised learning

Data-Xray draws upon many truly excellent python libraries, 
for which we as developers are phenomenally greatful.

