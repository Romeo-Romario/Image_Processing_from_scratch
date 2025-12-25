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

// constants
// Compute magnitude in x direction
const vector<vector<double>> dI_dx = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}};

// Compute magnitude in y direction
const vector<vector<double>> dI_dy = {
    {1, 2, 1},
    {0, 0, 0},
    {-1, -2, -1}};

// Laplacian matrix
const vector<vector<double>> d2I_dxdy = {
    {1 * (1.0 / 6.0), 4 * (1.0 / 6.0), 1 * (1.0 / 6.0)},
    {4 * (1.0 / 6.0), -20 * (1.0 / 6.0), 4 * (1.0 / 6.0)},
    {1 * (1.0 / 6.0), 4 * (1.0 / 6.0), 1 * (1.0 / 6.0)}};

std::vector<std::vector<double>>
gaussian_kernel(const double sigma);
void convolve_chunk(
    const std::vector<std::vector<double>> &grey_image,
    const std::vector<std::vector<double>> &kernel,
    std::vector<std::vector<double>> &result,
    int start_row, int end_row);

void chunk_zero_crossing(std::vector<std::vector<double>> &d2I, int start_row, int end_row);
void chunk_gradient_magnitute(const Matrix &dI_dX, const Matrix &dI_dY, Matrix &dI_magnitued, int start_row, int end_row);