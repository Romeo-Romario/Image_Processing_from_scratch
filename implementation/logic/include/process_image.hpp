#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<double> process_image(py::array_t<double> grey_image);