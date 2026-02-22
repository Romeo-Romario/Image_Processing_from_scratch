#include "./include/text_box_detector.hpp"
#include <pybind11/stl.h>
namespace py = pybind11;

PYBIND11_MODULE(TextBoxDetector, m)
{
     m.doc() = "This module implements logic of TextBoxDetector";

     py::class_<Point>(m, "Point")
         .def(py::init<>())
         .def(py::init<int, int>())
         .def_readwrite("x", &Point::x)
         .def_readwrite("y", &Point::y);

     py::class_<TextRow>(m, "TextRow")
         .def(py::init<>())
         .def_readwrite("text_matrix", &TextRow::text_matrix)
         .def_readwrite("y_start", &TextRow::y_start)
         .def_readwrite("y_end", &TextRow::y_end)
         .def_readwrite("x_start", &TextRow::x_start)
         .def_readwrite("x_end", &TextRow::x_end)
         .def_readwrite("index_in_original_img", &TextRow::index_in_original_img)
         .def_readwrite("_1d_function", &TextRow::_1d_function)
         .def_readwrite("zero_sep_points", &TextRow::zero_sep_points)
         .def_readwrite("potetional_zero_sep_points", &TextRow::potetional_zero_sep_points)
         .def_readwrite("symbols_limits", &TextRow::symbols_limits);

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
         .def("detect_symbol_boxes", &TextBoxDetector::detect_symbol_boxes, py::arg("pixel_threshold") = 3.0)
         .def("smooth_row_function", &TextBoxDetector::smooth_row_function);
}