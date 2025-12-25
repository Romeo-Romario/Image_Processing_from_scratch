#pragma once

#include <vector>
#include <cmath>
#include <numeric>
#include <thread>
#include <array>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using std::vector;
using Matrix = vector<vector<double>>;

std::vector<py::array_t<double>> convert_matrixes_pointers_to_numpy_array(const vector<Matrix *> &matrixes);
std::vector<py::array_t<double>> convert_matrixes_to_numpy_array(const vector<Matrix> &matrixes);