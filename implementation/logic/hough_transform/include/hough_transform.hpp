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
#include "./additional_functions.hpp"

namespace py = pybind11;
using std::vector;
using Matrix = vector<vector<double>>;

class HoughTransform
{
private:
    Matrix edges;
    double rho;
    double theta;
    double threshold;

    // Variables that will be used during computation process
    // Step 1:
    double diagonal;
    vector<double> theta_angles, rho_values;
    int num_thetas, num_rhos;
    Matrix accumulator;

public:
    HoughTransform() = default;
    HoughTransform(const HoughTransform &el);
    HoughTransform(const py::array_t<double> &input_edges, const double theta, const double rho);

    vector<Matrix> hough_lines(double threshold, double min_theta, double max_theta);
    double get_deskew_angle(double threshold, double min_theta, double max_theta);
};