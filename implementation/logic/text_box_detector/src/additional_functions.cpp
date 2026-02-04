#include <vector>
#include <cmath>
#include <numeric> // for std::accumulate if needed, though we sum manually below

std::vector<double> gaussian_kernel_1d(const double sigma)
{
    int size = static_cast<int>(2 * std::ceil(3 * sigma) + 1);
    int half = size / 2;

    std::vector<double> kernel(size);
    double sum = 0.0;

    for (int i = -half; i <= half; ++i)
    {
        double value = std::exp(-(i * i) / (2.0 * sigma * sigma));

        kernel[i + half] = value;
        sum += value;
    }

    for (int i = 0; i < size; ++i)
    {
        kernel[i] /= sum;
    }

    return kernel;
}