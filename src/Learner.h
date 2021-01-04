#ifndef PYGIF_LEARNER_H
#define PYGIF_LEARNER_H

#include <src/io/DataTypeHandling.h>

namespace pygif {
    /**
     * Provides a standardized interface for learning algorithms.
     */
    template<typename ModelType, typename PredictionType>
    class Learner {
    public:
        /**
         * Fits the learner using a given dataset. Classes implementing this method should assign the learned model
         * to themselves, which can then be retrieved using the getModel() method.
         *
         * @param dataset The dataset, which should be used for fitting.
         * @return A reference to the learner.
         */
        virtual Learner<ModelType, PredictionType>& fit(const MatrixX& dataset) {
            throw std::runtime_error("Learner::fit: Not implemented.");
        };

        /**
         * Fits the learner using a given dataset and returns predictions based on it. Classes implementing this
         * method should assign the learned model to themselves, which can then be retrieved using the getModel() method.
         *
         * The behaviour of this method should be equal to running the fit() and the predict() method.
         *
         * @param dataset The dataset, which should be used for fitting and predicting.
         * @return Predictions, which were made by using the dataset. The actual type of prediction, which is made, is determined by the PredictionType template parameter.
         */
        virtual PredictionType fitPredict(const MatrixX& dataset) {
            return fit(dataset).predict(dataset);
        };

        /**
         * Make predictions using a priorly fitted model and a given dataset. Classes implementing this method
         * should use the same model, which is given by the getModel() method and which is usually found by invoking
         * the fit() method.
         *
         * @param dataset The dataset, which should be used for predicting.
         * @return Predictions, which were made by using the dataset. The actual type of prediction, which is made, is determined by the PredictionType template parameter.
         */
        virtual PredictionType predict(const MatrixX& dataset) const {
            throw std::runtime_error("Learner::predict: Not implemented.");
        };

        /**
         * Make predictions using a priorly fitted model and a given dataset. Classes implementing this method
         * should use the same model, which is given by the getModel() method and which is usually found by invoking
         * the fit() method.
         *
         * @param dataset The dataset, which should be used for predicting.
         * @param model The fitted model.
         * @return Predictions, which were made by using the dataset. The actual type of prediction, which is made, is determined by the PredictionType template parameter.
         */
        virtual PredictionType predict(const MatrixX& dataset, const ModelType& model) const {
            throw std::runtime_error("Learner::predict: Not implemented.");
        };

        /**
         * Returns the currently fitted model.
         *
         * @return The currently fitted model. The actual type of model, which is returned by this method, is determined by the ModelType template parameter.
         */
        virtual ModelType getModel() const {
            throw std::runtime_error("Learner::getModel: Not implemented.");
        };

        /**
         * Takes a copy of the learner and returns it.
         * @return Non-const copy of the learner.
         */
        virtual std::unique_ptr<Learner<ModelType, PredictionType>> copy() const {
            throw std::runtime_error("Learner::copy: Not implemented.");
        }

        /**
         * Learners destructor.
         */
        virtual ~Learner() = default;
    };
}

#endif // PYGIF_LEARNER_H
