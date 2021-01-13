#ifndef GENIF_KERNEL_H
#define GENIF_KERNEL_H

#include "Kernel.h"

namespace genif {
    class Kernel {
    public:
        /**
         * Returns the value of the kernel function for two vectors.
         * @param x1 A real-valued vector.
         * @param x2 A real-valued vector.
         * @return The value of the kernel function k(x1, x2).
         */
        virtual data_t operator()(const VectorX& x1, const VectorX& x2) const = 0;

        /**
         * Destructor.
         */
        virtual ~Kernel() = default;
    };
}

#endif // GENIF_KERNEL_H
