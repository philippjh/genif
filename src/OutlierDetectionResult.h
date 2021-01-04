#ifndef GENIF_OUTLIERDETECTIONRESULT_H
#define GENIF_OUTLIERDETECTIONRESULT_H

#include <src/io/DataTypeHandling.h>

namespace genif {
    /**
     * A struct holding a outlier detection result.
     */
    struct OutlierDetectionResult {
        VectorX probabilities;

        /**
         * Returns a const-reference to the probability vector stored in this object. The particular notion of probability is determined by the algorithm used. GIF will usually
         * report higher probabilities to be indicating inlierness.
         * @return As stated above.
         */
        const VectorX& getProbabilities() const {
            return probabilities;
        }
    };
}

#endif // GENIF_OUTLIERDETECTIONRESULT_H
