#include "pyindex.h"

PYBIND11_MODULE(ccannpy, m) {
  m.doc() = "CCANN";
  m.attr("__version__") = "dev";

  py::enum_<ccann::Metric>(m, "Metric")
      .value("L2", ccann::Metric::L2)
      .value("COSINE", ccann::Metric::COSINE)
      .export_values();
}