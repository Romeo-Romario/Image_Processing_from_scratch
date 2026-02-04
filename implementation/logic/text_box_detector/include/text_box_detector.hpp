#pragma once
#include <vector>
#include <iostream>
#include <cmath>
#include <utility>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../../additional_modules/include/matrix_convert.hpp"
#include "../../additional_modules/include/threading.hpp"
#include "./additional_functions.hpp"

namespace py = pybind11;

using std::cout;
using std::endl;
using std::vector;

using Matrix = vector<vector<double>>;
#define M_PI 3.14159265358979323846

class TextBoxDetector
{
    Matrix deskew_canny_image;
    vector<double> smoothed_img_f;
    unsigned int n_threads = std::max(1u, std::thread::hardware_concurrency());

public:
    TextBoxDetector() = default;

    TextBoxDetector(const py::array_t<double> &deskew_canny_image);

    // TODO: remake
    vector<double> smooth_row_function();
    vector<bool> find_extream_points();
};