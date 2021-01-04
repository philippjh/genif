=====
Usage
=====

Once you have installed (for Python users) or downloaded (for C++ users) the package, the module can be used in two different ways, either by importing ``genif`` into Python or by
including the respective sources into your C++ project.

Python
======

The main functionality of this package is implemented in the :py:class:`genif.GeneralizedIsolationForest` class. For convenience, this class follows the well-established principle
of librarys like ``scikit-learn``, which requires classifiers to provide a ``fit``, ``fit_predict`` and ``predict`` method. A very basic example using this library can hence be
given as follows:

.. code-block:: python

    import numpy as np
    from genif import GeneralizedIsolationForest

    # Create some random demo data.
    N = 1000
    d = 50
    X = np.random.random((N, d))

    # Create the GIF classifier.
    gif = GeneralizedIsolationForest(k=10, n_models=50, sample_size=256,
                                     kernel="rbf", kernel_scaling=[0.05], sigma=0.01)

    # Fit the classifier and make predictions.
    y_pred = gif.fit_predict(X)

For this example, we chose to divide every data region into ``k=10``. The algorithm will fit a total of 50 trees, each considering a sample of the provided dataset containing 256
observations. To decide, when tree induction is terminated, the algorithm relies on a RBF kernel which is scales to 0.05. Tree induction is terminated, when the pairwise average
kernel value of observations in a particular exceeds 0.01.

The GIF algorithm internally fits a forest of Generalized Isolation Trees and estimates inlier probabilities for each found data region. After the procedure finished, the returned
values is assigned to ``y_pred``, which contains a vector of ``N = 1000`` entries each describing the probability for every input data vector to be inlying or not. High probability
values, which are near one, therefore indicate conforming (i.e. "normal") behaviour. Conversely, probability values near zero indicate non-conforming (i.e. "anomalous") behaviour.

It is also possible to fit the GIF on one dataset, while using another dataset for the actual predictions you want to make. In this case, you will need to call ``fit`` and
``predict`` independently:

.. code-block:: python

    import numpy as np
    from genif import GeneralizedIsolationForest

    # Create some random demo data (1).
    X_training = np.random.random((1000, 50))
    X_testing = np.random.random((10, 50))

    # Create the GIF classifier.
    gif = GeneralizedIsolationForest(k=10, n_models=50, sample_size=256,
                                     kernel="rbf", kernel_scaling=[0.05], sigma=0.01)

    # Fit the classifier and make predictions.
    y_pred = gif.fit(X_training).predict(X_testing)

You may also want to choose another kernel to check for tree induction termination. Besides the RBF kernel, the class of Matèrn kernels is supported with
:math:`\nu \in \left\lbrace 1/2, 3/2, 5/2 \right\rbrace`, which can be selected in code by replacing ``rbf`` with ``matern-d1``, ``matern-d3``, ``matern-d5`` respectively. Please
keep in mind, that the Matèrn kernels expect the scaling vector to contain as many entries as the input vectors have dimensions. Thus, GIF may be called like that:

.. code-block:: python

    import numpy as np
    from genif import GeneralizedIsolationForest

    # Create some random demo data.
    d = 50
    X = np.random.random((1000, d))

    # Create the GIF classifier.
    gif = GeneralizedIsolationForest(k=10, n_models=50, sample_size=256,
                                     kernel="rbf", kernel_scaling=np.repeat(0.5, d), sigma=0.01)

    # Fit the classifier and make predictions.
    y_pred = gif.fit_predict(X)

Remember that GIF returns probability values, which you want to be binarized. In this case you will need to find an appropriate probability threshold, which you can apply to the
prediction vector for binarization.

C++
===

Using the C++ interface might be interesting for those users, which want to embed this algorithm either in their existing programs or which want to add more functionality to the
existing sources (what we highly appreciate! Merge requests are always welcome.). For the C++ part of this section, we will discuss the general project
setup routine rather than the parametrization options for GIF. If you're interested in those, please take a look into the `Python` subsection above as the necessary parameters are
quite the same.

Using the library within other projects
---------------------------------------

The ``genif`` sources are distributed as a "header-only" library within the CMake project model. Hence, no explicit compilation or linking is needed. For this section, we will
assume, that your project is also organized as a CMake project.

To include GIF in your package follow these steps:

1. `Optional:` Create a separate subdirectory holding library folders (e.g. ``lib``).
2. **Recursively** clone GIF source code repository by issueing either ``git clone --recurse-submodules git@github.com:philippjh/genif.git`` or ``git submodule add --recurse-submodules git@github.com:philippjh/genif.git && git submodule update --init --recursive`` (for submodule enthusiasts).
3. Add the subdirectory to your ``CMakeLists.txt`` file (i.e. ``add_subdirectory(lib/genif)``).
4. Link "your" target to the "virtual" target ``libgenif``, which makes all necessary header files available to your project. This can be accomplished by ``target_link_libraries(yourtarget PUBLIC libgenif)``.

You are ready to use the GIF library within your C++ project. All GIF-related symbols are packed into the ``genif`` namespace, hence do not forget to either prepend ``genif::`` or
use an ``using namespace`` directive.

A short demonstrational listing may be given as follows:

.. code-block:: c++

    #include <iostream>
    #include <src/gif/GeneralizedIsolationForest.h>

    int main() {
        // Create some parameters.
        const unsigned int k = 10;
        const unsigned int nModels = 100;
        const unsigned int sampleSize = 256;
        const std::string kernelId = "rbf";
        const Eigen::VectorXd kernelScaling = Eigen::VectorXd::Random(1);
        const double sigma = 0.02;
        const int workerCount = -1;

        // Create some random data to classify.
        const unsigned int N = 1000;
        const unsigned int d = 50;
        auto X = Eigen::MatrixXd::Random(N, d);

        genif::GeneralizedIsolationForest gif(k, nModels, sampleSize, kernelId, kernelScaling, sigma, workerCount);
        auto yPred = gif.fitPredict(X);

        std::cout << "Prediction:" << std::endl << std::endl << yPred << std::endl;

        return 0;
    }

As you can see, GIF uses the Eigen library for matrix-vector operations, which is included automatically, when you add the library to your ``CMakeLists.txt``.