#include "../include/additional_functions.hpp"
#include <cmath>
#include <algorithm>
using std::signbit;

std::vector<std::vector<double>> gaussian_kernel(const double sigma)
{
    // Compute kernel size (ensure it's odd)
    int size = static_cast<int>(2 * std::ceil(3 * sigma) + 1);
    int half = size / 2;

    std::vector<std::vector<double>> kernel(size, std::vector<double>(size));
    double sum = 0.0;

    // Fill kernel values using the 2D Gaussian function
    for (int i = -half; i <= half; ++i)
    {
        for (int j = -half; j <= half; ++j)
        {
            double value = std::exp(-(i * i + j * j) / (2.0 * sigma * sigma));
            kernel[i + half][j + half] = value;
            sum += value;
        }
    }

    // Normalize kernel so that its sum equals 1
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            kernel[i][j] /= sum;
        }
    }

    return kernel;
}

void convolve_chunk(
    const std::vector<std::vector<double>> &grey_image,
    const std::vector<std::vector<double>> &kernel,
    std::vector<std::vector<double>> &result,
    int start_row, int end_row)
{
    int kernel_size = kernel.size();
    int kernel_half_size = kernel_size / 2;
    int rows = grey_image.size();
    int cols = grey_image[0].size();

    for (int row = start_row; row < end_row; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            double result_pixel_value = 0.0;

            for (int k_row = 0; k_row < kernel_size; ++k_row)
            {
                for (int k_col = 0; k_col < kernel_size; ++k_col)
                {
                    int src_row = row - kernel_half_size + k_row;
                    int src_col = col - kernel_half_size + k_col;

                    if (src_row < 0 || src_col < 0 || src_row >= rows || src_col >= cols)
                        continue;

                    result_pixel_value += grey_image[src_row][src_col] * kernel[k_row][k_col];
                }
            }

            result[row][col] = result_pixel_value;
        }
    }
}

void chunk_gradient_magnitute(const Matrix &dI_dX, const Matrix &dI_dY, Matrix &dI_magnitued, int start_row, int end_row)
{
    int col_size = dI_magnitued[0].size();
    for (int row = start_row; row < end_row; row++)
    {
        for (int col = 0; col < col_size; col++)
        {
            dI_magnitued[row][col] = std::pow(std::pow(dI_dX[row][col], 2) + std::pow(dI_dY[row][col], 2), 0.5);
        }
    }
}