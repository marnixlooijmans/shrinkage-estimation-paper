# Shrinkage Estimators of the Cosmological Precision Matrix 

This repository contains the initial power spectrum data, all generated data and the code used to generate these data belonging to the paper ([arXiv:2402.13783](https://arxiv.org/abs/2402.13783)).

The implementations of the shrinkage estimation algorithms can be found in /lib/shrinkage_estimators. The algorithms are from the following papers:
* NERCOME for covariance matrix, Joachimi B., 2017, MNRAS, 466, L83;
* linear shrinkage of covariance matrix, Pope A. C., Szapudi I., 2008, MNRAS, 389, 766;
* linear shrinkage of precision matrix, Bodnar T., Gupta A. K., Parolya N., 2016, J. Multivar. Anal., 146, 223.

All generated data are in /output.

In order to run all code in this repository, the following packages are required: [NumPy](https://numpy.org), [SciPy](https://scipy.org), [Matplotlib](https://matplotlib.org), [GetDist](https://getdist.readthedocs.io/en/latest/), [pocoMC](https://pocomc.readthedocs.io) and [nbodykit](https://nbodykit.readthedocs.io/).
