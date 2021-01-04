#ifndef PYGIF_RBFKERNEL_H
#define PYGIF_RBFKERNEL_H

#include "Kernel.h"

namespace pygif {
    class RBFKernel : public Kernel {
    public:
        using Kernel::operator();

        /**
         * The default constructor for this kernel. The sigma value is 1.0.
         */
        RBFKernel() = default;

        /**
         * Instantiates a RBF Kernel object with given sigma.
         * @param sigma Kernel sigma value.
         */
        explicit RBFKernel(data_t sigma) : RBFKernel(sigma, 1.0) {
        }

        /**
         * Instantiates a RBF Kernel object with given sigma.
         * @param sigma Kernel sigma value.
         * @param l A scaling value.
         */
        RBFKernel(data_t sigma, data_t l) : _l2(std::pow(l, 2.0)) {
            _sigma = sigma;
            _denom = 2.0 * std::pow(_sigma, 2.0);
        };

        /**
         * Returns the RBF kernel value for two vectors x1 and x2 by using the following formula.
         *
         * \f$k(x_1, x_2) = _l^2 \exp(- \frac{\|x_1 - x_2 \|_2^2}{2\sigma^2})\f$
         *
         * The norm in the equation may be substituted by a different function by constructing this object with another distance function.
         * However, the implementation defaults to the squared L2-norm, which effectively resembles the squared euclidean distance between
         * the input vectors.
         *
         * @param x1 A vector.
         * @param x2 A vector.
         * @return RBF kernel value for x1 and x2.
         */
        inline data_t operator()(const VectorX& x1, const VectorX& x2) const override {
            return _l2 * std::exp(-((x1 - x2).squaredNorm()) / _denom);
        }

        /**
         * Returns the sigma value, that is currently stored in this kernel.
         * @return As stated above.
         */
        data_t getSigma() const {
            return _sigma;
        }

        /**
         * Returns the L scaler value, that is currently stored in this kernel.
         * @return As stated above.
         */
        data_t getL() const {
            return sqrt(_l2);
        }

    private:
        data_t _sigma = 1.0;
        data_t _denom = 0.0;
        data_t _l2 = 1.0;
    };
}

#endif // PYGIF_RBFKERNEL_H
