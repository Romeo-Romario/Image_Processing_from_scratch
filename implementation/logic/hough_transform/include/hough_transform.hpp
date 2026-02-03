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
using std::vector;
using Matrix = vector<vector<double>>;

#define M_PI 3.14159265358979323846
extern unsigned int n_threads;

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
    Matrix get_rotation_matrix(std::pair<int, int> center, double angle, double scale);
    py::array_t<double> rotate_image(const Matrix &image, Matrix &rotation_matrix);
    py::array_t<double> deskew_image(const py::array_t<double> &image, double threshold, double min_theta, double max_theta);
};

// Additional method for image rotation
py::array_t<double> conditional_rotation(const py::array_t<double> &image);