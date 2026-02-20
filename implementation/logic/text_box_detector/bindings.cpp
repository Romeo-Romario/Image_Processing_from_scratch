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
         .def("find_extream_points", &TextBoxDetector::find_extream_points,
              py::arg("global_average_threshold") = 0.7,
              py::arg("mean_distance_threshold") = 0.8)
         .def("get_text_rows", &TextBoxDetector::get_text_rows)
         .def("seperate_main_text", &TextBoxDetector::seperate_main_text)
         .def("get_clean_text_rows", &TextBoxDetector::get_clean_text_rows)
         .def("remove_rows_without_text", &TextBoxDetector::remove_rows_without_text,
              py::arg("remove_threshold") = 8.0,
              py::arg("width_threshold") = 40)
         .def("smooth_row_function", &TextBoxDetector::smooth_row_function);
}