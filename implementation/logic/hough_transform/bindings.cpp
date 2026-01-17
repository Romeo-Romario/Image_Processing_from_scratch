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

        .def("hough_lines", &HoughTransform::hough_lines,
             py::arg("threshold"),
             py::arg("min_theta"),
             py::arg("max_theta"));
    // .def("get_hysteresis_img", &CannyEdgeDetector::get_hysteresis_img)
    // .def("get_canny_img", &CannyEdgeDetector::get_canny_img,
    //      py::arg("grey_img"),
    //      py::arg("sigma") = 1.0,
    //      py::arg("hight_threshold") = 0.4,
    //      py::arg("low_threshold") = 0.06);
}