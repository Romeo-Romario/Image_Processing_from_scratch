#include "./include/process_image.hpp"

PYBIND11_MODULE(geometry, m)
{
    m.doc() = "TODO: add docstring";
    m.def("process_image", &process_image, "Do something",
          py::arg("grey_image"));
}