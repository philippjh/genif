#ifndef GENIF_GENERALIZEDISOLATIONFOREST_H
#define GENIF_GENERALIZEDISOLATIONFOREST_H

#include "GeneralizedIsolationTree.h"
#include <genif/BaggingEnsemble.h>
#include <genif/Learner.h>
#include <genif/OutlierDetectionResult.h>
#include <genif/Tools.h>
#include <genif/gif/GIFModel.h>

namespace genif {
    class GeneralizedIsolationForest : public Learner<std::vector<GIFModel>, VectorX> {
    public:
        /**
         * Instantiates a GeneralizedIsolationForest.
         * @param k The number of representatives to find for each node of the tree.
         * @param nModels The number of trees to fit.
         * @param sampleSize The sample size to consider for every tree to be fit.
         * @param kernelId Name of the kernel to use (possible values: rbf, matern-d1, matern-d3, matern-d5).
         * @param kernelScaling Vector of scaling values for the kernel to be used (scalar for RBF, d-dimensional vector for Matern kernels - d being the number of dimensions of
         * the input vectors).
         * @param sigma Average pairwise kernel values of observations in a data sub-region, which should be exceeded for the exit condition to apply.
         * @param workerCount Number of parallel workers to consider (-1 defaults to all available cores).
         * @param seed Seed to use for random number generation (-1 defaults to sysclock seed). Pass an integer for constant result across multiple runs.
         */
        GeneralizedIsolationForest(unsigned int k, unsigned int nModels, unsigned int sampleSize, const std::string& kernelId, const VectorX& kernelScaling, data_t sigma,
                                   int workerCount = -1, int seed = -1) :
            _exitCondition(kernelId, kernelScaling, sigma),
            _gTree(k, _exitCondition, genif::Tools::handleWorkerCount(workerCount), seed), _gtrBagging(_gTree, nModels, sampleSize, genif::Tools::handleWorkerCount(workerCount), seed) {
        }

        /**
         * Fits all trees.
         * @param dataset The dataset to use for fitting.
         * @return A reference to this object.
         */
        Learner<std::vector<GIFModel>, VectorX>& fit(const MatrixX& dataset) override {
            _gtrBagging.fit(dataset);
            return *this;
        }

        /**
         * Predicts the outlierness of a dataset by inspecting the learned forest of trees.
         * @param dataset The dataset to inspect.
         * @return A vector, which indicates the probability of inlierness for every input vector.
         */
        VectorX predict(const MatrixX& dataset) const override {
            if(_gtrBagging.getActualNumberOfModels() > 0) {
                // Get predictions.
                const std::vector<OutlierDetectionResult>& predictions = _gtrBagging.predict(dataset);

                // Average over predictions.
                VectorX y(dataset.rows());
                for (unsigned int i = 0; i < dataset.rows(); i++) {
                    data_t predictionSum = 0.0;
                    for (unsigned int j = 0; j < _gtrBagging.getActualNumberOfModels(); j++)
                        predictionSum += predictions[j].getProbabilities()[i];
                    y[i] = predictionSum / static_cast<data_t>(_gtrBagging.getActualNumberOfModels());
                }
                return y;
            } else throw std::runtime_error("GeneralizedIsolationForest::predict: Number of models is insufficient (maybe forgot to call `fit`?).");
        }

        /**
         * Returns the learned vector of GIFModels i.e. the trees.
         * @return As stated above.
         */
        std::vector<GIFModel> getModel() const override {
            return _gtrBagging.getModel();
        }

        /**
         * Destructor.
         */
        ~GeneralizedIsolationForest() override = default;

    private:
        GIFExitConditionAverageKernelValue _exitCondition;
        GeneralizedIsolationTree _gTree;
        BaggingEnsemble<GIFModel, OutlierDetectionResult> _gtrBagging;
    };
}

#endif // GENIF_GENERALIZEDISOLATIONFOREST_H
