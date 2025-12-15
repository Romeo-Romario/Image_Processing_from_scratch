#include "../include/additional_functions.hpp"

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