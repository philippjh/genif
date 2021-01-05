#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <src/gif/GeneralizedIsolationForest.h>
#include <src/io/DataTypeHandling.h>

namespace py = pybind11;

namespace genif {
    PYBIND11_MODULE(genif, m) {
        // Definition: Generalized Isolation Forest
        using GIFModel_VecX_Learner = Learner<std::vector<GIFModel>, VectorX>;
        py::class_<GIFModel_VecX_Learner>(m, "GIFModel_ODR_Learner");
        py::class_<GeneralizedIsolationForest, GIFModel_VecX_Learner>(m, "GeneralizedIsolationForest")
            .def(py::init<unsigned int, unsigned int, unsigned int, std::string, VectorX&, data_t, int, int>(), py::arg("k"), py::arg("n_models"), py::arg("sample_size"),
                 py::arg("kernel"), py::arg("kernel_scaling"), py::arg("sigma"), py::arg("worker_count") = -1, py::arg("seed") = -1)
            .def("fit", &GeneralizedIsolationForest::fit, py::arg("X"))
            .def("predict", &GeneralizedIsolationForest::predict, py::arg("X"))
            .def("fit_predict", &GeneralizedIsolationForest::fitPredict, py::arg("X"))
            .def_property_readonly("models", &GeneralizedIsolationForest::getModel);
    }
}