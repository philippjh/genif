#ifndef GENIF_BAGGINGENSEMBLE_H
#define GENIF_BAGGINGENSEMBLE_H

#include <chrono>
#include <random>
#include <src/Learner.h>

namespace genif {
    /**
     * Provides a generic Bagging ensemble, that randomly subsamples given datasets and outputs a vector of learned models and predictions.
     *
     * @tparam ModelType The type of model, that the learner yields as result from calling `fit`.
     * @tparam PredictionType The type of prediction, that the learner yields as result from calling either `fitPredict` and `predict`.
     */
    template<typename ModelType, typename PredictionType>
    class BaggingEnsemble : public Learner<std::vector<ModelType>, std::vector<PredictionType>> {
    public:
        /**
         * Constructs a new instance of BaggingEnsemble.
         *
         * @param baseLearner The base learner, which serves as a prototype for subsequent learning efforts.
         * @param nModels The number of models to fit.
         * @param sampleSize The number of observations to draw to fit each model.
         * @param workerCount The number of workers, which should fit models in parallel.
         * @param seed Seed to use for random number generation (-1 defaults to sysclock seed). Pass an integer for constant result across multiple runs.
         */
        explicit BaggingEnsemble(const Learner<ModelType, PredictionType>& baseLearner, unsigned int nModels = 100, unsigned int sampleSize = 256, unsigned int workerCount = 1,
                                 int seed = -1) :
            _baseLearner(baseLearner), _seed(seed) {
            // Check property validity.
            if (nModels <= 0)
                throw std::runtime_error("BaggingEnsemble::BaggingEnsemble: nModels needs to be greater than zero.");
            if (sampleSize <= 0)
                throw std::runtime_error("BaggingEnsemble::BaggingEnsemble: sampleSize needs to be greater than zero.");
            if (workerCount <= 0)
                throw std::runtime_error("BaggingEnsemble::BaggingEnsemble: workerCount needs to be greater than zero.");

            // Assign properties.
            _nModels = nModels;
            _sampleSize = sampleSize;
            _workerCount = workerCount;
        }

        /**
         * Fit `nModels` using the supplied dataset.
         *
         * For fitting, this method takes a copy of the baseLearner property and calls fit using a previously drawn sample of dataset.
         *
         * @param dataset The dataset used to fit models.
         * @return A reference to the current BaggingEnsemble instance. The fitted models may be retrieved by calling the `getModels()` function.
         */
        Learner<std::vector<ModelType>, std::vector<PredictionType>>& fit(const MatrixX& dataset) override {
            // Create PRNG.
            std::default_random_engine generator(_seed >= 0 ? _seed : std::chrono::system_clock::now().time_since_epoch().count());
            std::uniform_int_distribution<int> distribution(0, dataset.rows() - 1);

            // Remove all existing models.
            _models.clear();

            // Estimate new models.
#pragma omp parallel for num_threads(_workerCount)
            for (unsigned int i = 0; i < _nModels; i++) {
                // Take a copy of the base learner.
                auto learnerCopy = _baseLearner.copy();

                // Sample dataset with replacement.
                MatrixX sampledDataset(_sampleSize, dataset.cols());
                for (unsigned int j = 0; j < _sampleSize; j++)
                    sampledDataset.row(j) = dataset.row(distribution(generator));

                // Fit base learner with sampled dataset.
                learnerCopy->fit(sampledDataset);

                // Add estimated model to the models vector.
#pragma omp critical
                _models.push_back(std::move(learnerCopy->getModel()));
            }

            // Return self.
            return *this;
        }

        /**
         * Make predictions by using the set of models, which were previously learned with the `fit` method.
         *
         * Internally, this method calls the predict method of the given base learner and calls predict with every learned model and the supplied dataset.
         *
         * @param dataset The dataset to use for prediction.
         * @return A vector of predictions.
         */
        std::vector<PredictionType> predict(const MatrixX& dataset) const override {
            // Create vector of predictions.
            std::vector<PredictionType> predictions;
            predictions.reserve(_models.size());

            // Make predictions from models.
            for (auto& model : _models)
                predictions.push_back(_baseLearner.predict(dataset, model));

            return predictions;
        }

        /**
         * Gathers the list of learned models, which were previously learned with the fit method.
         * @return A list of models.
         */
        std::vector<ModelType> getModel() const {
            return _models;
        }

        /**
         * Returns the number of models, which should be fitted in this ensemble.
         * @return As stated above.
         */
        unsigned int getNumberOfModels() const {
            return _nModels;
        }

        /**
         * Returns the number of models, which have been fitted in this ensemble.
         * @return As stated above.
         */
        unsigned int getActualNumberOfModels() const {
            return _models.size();
        }

    private:
        const Learner<ModelType, PredictionType>& _baseLearner;
        unsigned int _nModels;
        unsigned int _sampleSize;
        unsigned int _workerCount;
        int _seed;

        std::vector<ModelType> _models;
    };
}

#endif // GENIF_BAGGINGENSEMBLE_H
