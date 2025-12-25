#pragma once
#include <thread>
#include <iostream>
#include <vector>
#include <utility>
#include <memory>
#include "additional_functions.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using std::cout;
using std::endl;
using std::vector;
using Matrix = vector<vector<double>>;

class EdgeDetector
{
private:
    int rows, cols;
    vector<vector<double>> grey_image;
    vector<vector<double>> kernel_matrix;
    vector<vector<double>> convolved_image;
    vector<vector<double>> dI_dX, dI_dY, gradient_magnitued;

    int chunk_size;
    int n_threads = std::thread::hardware_concurrency();

public:
    EdgeDetector() : rows(0), cols(0), grey_image({}), convolved_image({}) {}
    EdgeDetector(const EdgeDetector &el);
    EdgeDetector(const vector<vector<double>> &input_image);
    EdgeDetector(const py::array_t<double> &input_image);

    vector<vector<double>> convolve_image(double sigma, bool output = false);
    vector<py::array_t<double>> get_image_gradients();
    vector<py::array_t<double>> get_image_gradient_orientation();
};