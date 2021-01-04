# Generalized Isolation Forest

This repository provides an Python implementation of the "Generalized Isolation Forest" (GIF) algorithm for unsupervised detection of outliers in data. 
GIF has originally been proposed in:

> Buschjäger, S., Honysz, PJ. & Morik, K. Randomized outlier detection with trees. International Journal of Data Science and Analytics (2020). https://doi.org/10.1007/s41060-020-00238-w

## Install from the Python Package Index (PyPI, recommended)

We provide Linux wheel packages for various Python versions, which can be installed like this:

```
pip install genif
```

Windows or macOS builds are currently **not** provided. Please resort to installation from source, if you are either using Windows or macOS.

## Install from source

### Requirements:

- GCC >= 5.4.0 (older versions or other compilers such as Clang or ICC may work, but have not been tested yet.)
- CMake >= 3.5.1
- OpenMP

### Build steps

- Recursively clone this repository by issueing `git clone --recurse-submodules git@github.com:philippjh/genif.git`
- Change your working directory to the root of the repository. Run `pip3 install .`
- The Python package manager will now build and install the package.