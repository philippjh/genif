#ifndef PYGIF_DATATYPEHANDLING_H_
#define PYGIF_DATATYPEHANDLING_H_

#include "Eigen/Eigen"
#include <cfloat>
#include <memory>

namespace pygif {
    typedef double data_t;
    typedef Eigen::MatrixXd MatrixX;
    typedef Eigen::VectorXd VectorX;
    typedef Eigen::Ref<VectorX, 0, Eigen::InnerStride<>> VectorXRef;
}

#endif
