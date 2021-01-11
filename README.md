# Generalized Isolation Forest [![Read Manual](https://img.shields.io/badge/read-manual-informational)](https://philippjh.github.io/genif/) [![PyPI](https://img.shields.io/pypi/v/genif)](https://pypi.org/project/genif/) [![PyPI - Format](https://img.shields.io/pypi/format/genif)](https://pypi.org/project/genif/)

This repository provides an Python implementation of the "Generalized Isolation Forest" (GIF) algorithm for unsupervised detection of outliers in data. 
GIF has originally been proposed in:

> Buschjäger, S., Honysz, PJ. & Morik, K. Randomized outlier detection with trees. International Journal of Data Science and Analytics (2020). https://doi.org/10.1007/s41060-020-00238-w

More information on this package, including a quick start guide, examples and how to use this within C++, is given [here](https://philippjh.github.io/genif/).

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

## Acknowledgments

Part of the work on this paper has been supported by Deutsche Forschungsgemeinschaft (DFG) within the Collaborative Research Center SFB 876 "Providing Information by 
Resource-Constrained Analysis", project A1, http://sfb876.tu-dortmund.de and by the German Competence Center for Machine Learning Rhine Ruhr 
(ML2R, https://www.ml2r.de, 01IS18038A), funded by the German Federal Ministry for Education and Research.  