#pragma once

#include <vector>
#include <cmath>
#include <numeric>
#include <thread>
#include <array>
#include <iostream>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#define M_PI 3.14159265358979323846

namespace py = pybind11;
using std::vector;
using Matrix = vector<vector<double>>;

// constants
// Compute magnitude in x direction
const Matrix dI_dx = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}};

// Compute magnitude in y direction
const Matrix dI_dy = {
    {1, 2, 1},
    {0, 0, 0},
    {-1, -2, -1}};

// Laplacian matrix
const Matrix d2I_dxdy = {
    {1 * (1.0 / 6.0), 4 * (1.0 / 6.0), 1 * (1.0 / 6.0)},
    {4 * (1.0 / 6.0), -20 * (1.0 / 6.0), 4 * (1.0 / 6.0)},
    {1 * (1.0 / 6.0), 4 * (1.0 / 6.0), 1 * (1.0 / 6.0)}};

// Map of directions for non-maximal suppression

const vector<std::pair<int, int>> direction_offsets = {
    {0, 1},
    {-1, 1},
    {-1, 0},
    {-1, -1},
    {0, -1},
    {1, -1},
    {1, 0},
    {1, 1}};

const double STRONG_PIXEL = 255.0;
const double WEAK_PIXEL = 50.0;

Matrix
gaussian_kernel(const double sigma);
void convolve_chunk(
    const Matrix &grey_image,
    const Matrix &kernel,
    Matrix &result,
    int start_row, int end_row);

void chunk_zero_crossing(Matrix &d2I, int start_row, int end_row);
void chunk_gradient_magnitute(const Matrix &dI_dX, const Matrix &dI_dY, Matrix &dI_magnitued, int start_row, int end_row);

Matrix calculate_gradient_orientation(const Matrix &dI_dX, const Matrix &dI_dY, int rows, int cols, int n_threads);
Matrix calculate_rounded_gradient(const Matrix &grad_oreo, double roundval, int rows, int n_threads);

inline std::pair<int, int> get_direction_index(double angle);

Matrix non_max_suppresion(const Matrix &rounded_grad_oreo, const Matrix &gradient_magnitued, int n_threads);
Matrix non_max_threshold(const Matrix &non_max_suppr_img, double maxx, double minn, double meann, int n_threads);
Matrix hysteresis(const Matrix &thresholded_img);