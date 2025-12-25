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
        .def("get_image_gradients", &EdgeDetector::get_image_gradients);
}