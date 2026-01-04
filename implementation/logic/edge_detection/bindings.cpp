#include "./include/edge_detector.hpp"
namespace py = pybind11;

PYBIND11_MODULE(EdgeDetector, m)
{
    m.doc() = "This Libary contains CannyEdgeDetector class. Main method of a class after initialization is get_canny_img which takes numpy two dimentional array as input";

    // Define CannyEdgeDetector class
    py::class_<CannyEdgeDetector>(m, "CannyEdgeDetector")
        // Bind the constructor that takes a numpy array
        .def(py::init<const py::array_t<double> &>())

        // (Optional) If you also want to expose the default constructor:
        .def(py::init<>())

        .def("get_convolved_image", &CannyEdgeDetector::get_convolved_image)
        .def("get_image_gradients", &CannyEdgeDetector::get_image_gradients)
        .def("get_image_gradient_orientation", &CannyEdgeDetector::get_image_gradient_orientation)
        .def("get_non_max_suppresion", &CannyEdgeDetector::get_non_max_suppresion)
        .def("get_thresholded_img", &CannyEdgeDetector::get_thresholded_img)
        .def("get_hysteresis_img", &CannyEdgeDetector::get_hysteresis_img)
        .def("get_canny_img", &CannyEdgeDetector::get_canny_img,
             py::arg("grey_img"),
             py::arg("sigma") = 1.0,
             py::arg("hight_threshold") = 0.4,
             py::arg("low_threshold") = 0.06);
}