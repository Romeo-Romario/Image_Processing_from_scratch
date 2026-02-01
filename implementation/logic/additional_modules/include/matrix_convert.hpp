#pragma once

#include <vector>
#include <cmath>
#include <numeric>
#include <thread>
#include <array>
#include <stdexcept>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using std::vector;
using Matrix = vector<vector<double>>;
using Matrix3D = vector<vector<vector<double>>>;

namespace additional_modules::matrix_converter
{
    std::vector<py::array_t<double>> convert_matrixes_pointers_to_numpy_array(const vector<Matrix *> &matrixes);
    std::vector<py::array_t<double>> convert_matrixes_to_numpy_array(const vector<Matrix> &matrixes);
    py::array_t<double> convert_matrix_to_numpy_array(const Matrix &matrix);
    Matrix convert_numpy_array_to_matrix(const py::array_t<double> &numpy_matrix);
    Matrix3D convert_numpy_array_to_3d_matrix(const py::array_t<double> &numpy_array);
}
