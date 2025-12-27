#include "./include/process_image.hpp"
#include "./include/edge_detector.hpp"
namespace py = pybind11;

PYBIND11_MODULE(geometry, m)
{
    m.doc() = "TODO: add docstring";
    m.def("process_image", &process_image, "Do something",
          py::arg("grey_image"));

    // Define EdgeDetector class
    py::class_<EdgeDetector>(m, "EdgeDetector")
        // Bind the constructor that takes a numpy array
        .def(py::init<const py::array_t<double> &>())

        // (Optional) If you also want to expose the default constructor:
        .def(py::init<>())

        .def("convolve_image", &EdgeDetector::convolve_image)
        .def("get_image_gradients", &EdgeDetector::get_image_gradients)
        .def("get_image_gradient_orientation", &EdgeDetector::get_image_gradient_orientation)
        .def("get_non_max_suppresion", &EdgeDetector::get_non_max_suppresion)
        .def("get_thresholded_img", &EdgeDetector::get_thresholded_img, py::arg("maxx"), py::arg("minn"), py::arg("meann"));
}