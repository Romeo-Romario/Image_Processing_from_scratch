#include "./include/hough_transform.hpp"
#include <pybind11/stl.h>
namespace py = pybind11;

PYBIND11_MODULE(HoughTransform, m)
{
     m.doc() = "This module implements logic of Hough Transform on previously determined edge map";

     // Define HoughTransform class
     py::class_<HoughTransform>(m, "HoughTransform")
         // Bind the constructor that takes a numpy array
         .def(py::init<const py::array_t<double> &, double, double>(),
              py::arg("input_edges"),
              py::arg("theta") = 0.261,
              py::arg("rho") = 9)

         .def("deskew_image", &HoughTransform::deskew_image, py::arg("image"), py::arg("threshold") = 2000.0,
              py::arg("min_theta") = -3.14159265358979323846,
              py::arg("max_theta") = 3.14159265358979323846);
}