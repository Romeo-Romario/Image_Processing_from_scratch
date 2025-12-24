#pragma once

#include <vector>
#include <cmath>
#include <numeric>
#include <thread>
#include <array>

using std::vector;
// constants
const vector<vector<double>> dI_dx = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}};

const vector<vector<double>> dI_dy = {
    {1, 2, 1},
    {0, 0, 0},
    {-1, -2, -1}};

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
