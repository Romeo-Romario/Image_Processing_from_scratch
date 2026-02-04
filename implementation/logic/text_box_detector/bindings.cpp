#include "./include/text_box_detector.hpp"
#include <pybind11/stl.h>
namespace py = pybind11;

PYBIND11_MODULE(TextBoxDetector, m)
{
    m.doc() = "This module implements logic of TextBoxDetector";

    py::class_<TextBoxDetector>(m, "TextBoxDetector")
        .def(py::init<>())
        .def(py::init<const py::array_t<double> &>(),
             py::arg("deskew_canny_image"))
        .def("find_extream_points", &TextBoxDetector::find_extream_points)
        .def("smooth_row_function", &TextBoxDetector::smooth_row_function);
}