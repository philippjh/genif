#ifndef GENIF_GIF_GIFMODEL_H
#define GENIF_GIF_GIFMODEL_H

#include <nanoflann.hpp>

namespace genif {
    /**
     * A struct comprising the model information for Generalized Isolation Forests.
     */
    struct GIFModel {
        std::vector<data_t> probabilitiesPerRegion;
        std::vector<unsigned long> countsPerRegion;
        std::shared_ptr<MatrixX> dataMatrix;
        std::shared_ptr<nanoflann::KDTreeEigenMatrixAdaptor<MatrixX>> dataKDTree;

        /**
         * Returns a vector of probabiltities for each found region (higher probability indicate inlierness).
         * @return As stated above.
         */
        const std::vector<data_t>& getProbabilitiesPerRegion() const {
            return probabilitiesPerRegion;
        };

        /**
         * Returns a vector of values, indicating how many vectors of the training data set fell into a particular region.
         * @return As stated above.
         */
        const std::vector<unsigned long>& getCountsPerRegion() const {
            return countsPerRegion;
        };
    };
}

#endif // GENIF_GIF_GIFMODEL_H
