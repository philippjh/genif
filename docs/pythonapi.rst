==========
Python API
==========

.. py:module:: genif
.. autoclass:: GeneralizedIsolationForest

    :members: models

    .. automethod:: __init__

        Initializes the GeneralizedIsolationForest with the following parameters:

        :param int k: The number of representatives to find for each node of the tree.
        :param int n_models: The number of trees to fit.
        :param int sample_size: The sample size to consider for every tree to be fit.
        :param str kernel: Name of the kernel to use (possible values: `rbf`, `matern-d1`, `matern-d3`, `matern-d5`).
        :param ndarray kernel_scaling: Vector of scaling values for the kernel to be used (scalar for RBF, ``d``-dimensional vector for Matern kernels).
        :param float sigma: Average pairwise kernel values of observations in a data sub-region, which should be exceeded for the exit condition to apply.
        :param int worker_count: Number of parallel workers to consider (-1 defaults to all available cores).

    .. automethod:: fit

        Fits the forest using the provided input data matrix.

        :param ndarray X:  Input data matrix with shape ``[n, d]``.
        :return: Callee.

    .. automethod:: fit_predict

        Fits the forest using the given input data matrix and predicts the probability for every input observation to be an inlier.

        :param ndarray X:  Input data matrix with shape ``[n, d]``.
        :return: Vector of probabilities, represented as ndarray with shape ``[n, 1]``.

    .. automethod:: predict

        Predicts the probability for inlierness for every entry of the data matrix. Prior to calling ``predict`` either ``fit`` or ``fit_predict`` has to be called.

        :param ndarray X:  Input data matrix with shape ``[n, d]``.
        :return: Vector of probabilities, represented as ndarray with shape ``[n, 1]``.
