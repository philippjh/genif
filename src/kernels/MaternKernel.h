#ifndef PYGIF_MATERNKERNEL_H
#define PYGIF_MATERNKERNEL_H

#include "Kernel.h"

namespace pygif {
    /**
     * Provides an implementation of the Matern kernel.
     */
    class MaternKernel : public Kernel {
    public:
        using Kernel::operator();

        /**
         * Instantiates the Matern kernel.
         * @param Sigma The scaling vector, which needs to hold as many entries as the vectors you want to compute the kernel function value with.
         * @param d
         * @param l
         */
        explicit MaternKernel(VectorX Sigma, unsigned int d = 3, data_t l = 1.0) : _d(d), _l(l), _Sigma(std::move(Sigma)) {
            if (!(d == 1 || d == 3 || d == 5))
                throw std::runtime_error("MaternKernel::MaternKernel: Only d=1 or d=3 or d=5 is supported.");
        }

        /**
         * Computes the Matern kernel function value for two vectors.
         * @param x1 A real-valued vector.
         * @param x2 A real-valued vector.
         * @return The Matern kernel function value \f$k(\vec{x}_1, \vec{x}_1)\f$.
         */
        data_t operator()(const VectorX& x1, const VectorX& x2) const override {
            if (_Sigma.size() != x1.size() || _Sigma.size() != x2.size())
                throw std::runtime_error("MaternKernel::operator(): The scaling vector size does not conform to the input vector dimensionalities.");
            VectorX x1Scaled = x1;
            VectorX x2Scaled = x2;
            for (unsigned int i = 0; i < _Sigma.size(); i++) {
                x1Scaled[i] = x1Scaled[i] / _Sigma[i];
                x2Scaled[i] = x2Scaled[i] / _Sigma[i];
            }

            data_t K = (x1Scaled - x2Scaled).norm() * sqrt(_d);

            if (_d == 1)
                return pow(_l, 2.0) * exp(-K);
            else if (_d == 3)
                return pow(_l, 2.0) * (1.0 + K) * std::exp(-K);
            else if (_d == 5)
                return pow(_l, 2.0) * (1.0 + K + pow(K, 2.0) / 3.0) * exp(-K);
            else
                throw std::runtime_error("MaternKernel::operator(): Only d=1 or d=3 or d=5 is supported.");
        }

        /**
         * Returns the `D` property of this Matern kernel instance.
         * @return As stated above.
         */
        unsigned int getD() const {
            return _d;
        }

        /**
         * Returns the `L` property of this Matern kernel instance.
         * @return As stated above.
         */
        data_t getL() const {
            return _l;
        }

        /**
         * Returns the scaling vector of this Matern kernel instance.
         * @return As stated above.
         */
        const VectorX& getSigma() const {
            return _Sigma;
        }

    private:
        unsigned int _d = 3;
        data_t _l = 1.0;
        VectorX _Sigma;
    };
}

#endif // PYGIF_MATERNKERNEL_H
