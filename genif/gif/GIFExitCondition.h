#ifndef GENIF_GIFEXITCONDITION_H
#define GENIF_GIFEXITCONDITION_H

#include "Tree.h"
#include <genif/kernels/Kernel.h>
#include <genif/kernels/MaternKernel.h>
#include <genif/kernels/RBFKernel.h>

namespace genif {
    /**
     * This class provides an interface to describe different exit conditions.
     */
    class GIFExitCondition {
    public:
        /**
         * Classes, which override this method, should return for a given node, whether a further recursion step should happen.
         *
         * More specifically, the parametrized node is subject to a split in a next recursion step, so this method decides,
         * whether this split should happen.
         *
         * @param node The node to make the decision for.
         * @return A decision, whether the next recursion step should happen.
         */
        virtual bool shouldExitRecursion(const Tree& node) const = 0;
    };

    class GIFExitConditionAverageKernelValue : public GIFExitCondition {
    public:
        GIFExitConditionAverageKernelValue(const GIFExitConditionAverageKernelValue&) = delete;
        GIFExitConditionAverageKernelValue& operator=(const GIFExitConditionAverageKernelValue&) = delete;

        /**
         * Initializes an exit condition-decider using average kernel values in a specific data subregion.
         * @param kernelId Name of the kernel to use (possible value: rbf, matern-d1, matern-d3, matern-d5).
         * @param kernelScaling Vector of scaling values for the kernel to be used (scalar for RBF, d-dimensional vector for Matern kernels - d being the number of dimensions of
         * the input vectors).
         * @param sigma Average kernel value, which should be exceeded for the exit condition to apply.
         */
        explicit GIFExitConditionAverageKernelValue(const std::string& kernelId, const VectorX& kernelScaling, data_t sigma) : _sigma(sigma) {
            if (kernelId == "rbf") {
                _kernel = new RBFKernel(kernelScaling[0]);
            } else if (kernelId == "matern-d1") {
                _kernel = new MaternKernel(kernelScaling, 1);
            } else if (kernelId == "matern-d3") {
                _kernel = new MaternKernel(kernelScaling, 3);
            } else if (kernelId == "matern-d5") {
                _kernel = new MaternKernel(kernelScaling, 5);
            } else {
                throw std::runtime_error("GIFExitConditionAverageKernelValue::GIFExitConditionAverageKernelValue: Unknown kernel supplied ('" + kernelId
                                         + "'). "
                                           "Possible choices are: rbf, matern-d1, matern-d3, matern-d5.");
            }
        };

        /**
         * Tests, whether a node should be subject to another recursion step.
         *
         * Instances of GIFExitConditionAverageKernelValue, check, whether the average kernel function value of the vectors w.r.t to the representative
         * in the node is greater than the specified sigma value and will only return true when this condition has been met.
         */
        bool shouldExitRecursion(const Tree& node) const override {
            auto* accuArray = new data_t[node.vectorIndices.size()];
            for (unsigned int i = 0; i < node.vectorIndices.size(); i++)
                accuArray[i] = _kernel->operator()(node.dataset.row(node.representativeIndex), node.dataset.row(node.vectorIndices[i]));

            data_t accu = 0.0;
#pragma omp simd reduction(+ : accu)
            for (unsigned int i = 0; i < node.vectorIndices.size(); i++)
                accu += accuArray[i];

            delete[] accuArray;
            return accu / static_cast<data_t>(node.vectorIndices.size()) >= _sigma;
        }

        /**
         * Destructor.
         */
        virtual ~GIFExitConditionAverageKernelValue() {
            delete _kernel;
        }

    private:
        Kernel* _kernel;
        data_t _sigma = 1.0;
    };
}

#endif // GENIF_GIFEXITCONDITION_H
