#include "../include/process_image.hpp"

py::array_t<double> process_image(py::array_t<double> grey_image)
{
    // Convert np array to vector<vector<double>>
    auto buf = grey_image.request();
    if (buf.ndim != 2)
        throw std::runtime_error("Input image must be 2D");

    size_t rows = buf.shape[0];
    size_t cols = buf.shape[1];
    auto ptr = static_cast<double *>(buf.ptr);

    std::vector<std::vector<double>> image(rows, std::vector<double>(cols));
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            image[i][j] = ptr[i * cols + j];

    std::vector<std::vector<double>> res = image;

    // Convert result back to numpy array
    py::array_t<double> output({rows, cols});
    auto out_buf = output.request();
    double *out_ptr = static_cast<double *>(out_buf.ptr);

    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            out_ptr[i * cols + j] = res[i][j];

    return output;
}