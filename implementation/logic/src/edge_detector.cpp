#include "../include/edge_detector.hpp"
#include "../include/matrix_converter.hpp"
#include "../include/threading.hpp"

EdgeDetector::EdgeDetector(const EdgeDetector &el)
{
    grey_image = el.grey_image;
    kernel_matrix = el.kernel_matrix;
    convolved_image = el.convolved_image;
    dI_dX = el.dI_dX;
    dI_dY = el.dI_dY;
    gradient_magnitued = el.gradient_magnitued;
}
EdgeDetector::EdgeDetector(const vector<vector<double>> &input_image)
{
    grey_image = input_image;
    rows = grey_image.size();
    cols = grey_image[0].size();
    chunk_size = rows / n_threads;
}
EdgeDetector::EdgeDetector(const py::array_t<double> &input_image)
{
    auto buf = input_image.request();
    if (buf.ndim != 2)
        throw std::runtime_error("Input image must be 2D");
    rows = buf.shape[0];
    cols = buf.shape[1];
    auto ptr = static_cast<double *>(buf.ptr);
    grey_image = vector(rows, std::vector<double>(cols));
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            grey_image[i][j] = ptr[i * cols + j];
    chunk_size = rows / n_threads;
}

vector<vector<double>> EdgeDetector::convolve_image(double sigma, bool output)
{
    // Set new Kernel matrix
    kernel_matrix = gaussian_kernel(sigma);
    if (output)
    {
        cout << "Kernel matrix\n";
        for (int i = 0; i < kernel_matrix.size(); i++)
        {
            for (int j = 0; j < kernel_matrix[i].size(); j++)
            {
                cout << kernel_matrix[i][j] << " ";
            }
            cout << endl;
        }
    }
    convolved_image = vector(rows, std::vector<double>(cols, 0.0));

    threading::split_to_threads(rows, n_threads, convolve_chunk, std::ref(grey_image), std::ref(kernel_matrix), std::ref(convolved_image));
    return convolved_image;
}

vector<py::array_t<double>> EdgeDetector::get_image_gradients()
{
    /*
        Based on Sobel 3x3 matrixes calculates dI/dx, dI/dy and magnitued matrixes
    */
    std::vector<Matrix *> matrix_ptrs = {&dI_dX, &dI_dY, &gradient_magnitued};
    vector<Matrix> derivative_matrixes = {dI_dx, dI_dy};

    for (int matrix_index = 0; matrix_index < matrix_ptrs.size() - 1; matrix_index++)
    {
        if (matrix_ptrs[matrix_index]->empty())
        {
            matrix_ptrs[matrix_index]->resize(rows, std::vector<double>(cols, 0.0));
        }

        threading::split_to_threads(rows, n_threads, convolve_chunk, std::ref(convolved_image),
                                    std::ref(derivative_matrixes[matrix_index]),
                                    std::ref(*matrix_ptrs[matrix_index]));
    }

    if (matrix_ptrs[2]->empty())
    {
        matrix_ptrs[2]->resize(rows, std::vector<double>(cols, 0.0));
    }

    threading::split_to_threads(rows, n_threads, chunk_gradient_magnitute,
                                std::ref(*matrix_ptrs[0]),
                                std::ref(*matrix_ptrs[1]),
                                std::ref(*matrix_ptrs[2]));

    std::vector<py::array_t<double>> result = convert_matrixes_pointers_to_numpy_array(std::move(matrix_ptrs));
    return result;
}

vector<py::array_t<double>> EdgeDetector::get_image_gradient_orientation()
{
    double roundval = M_PI / 4.0;
    grad_oreo = calculate_gradient_orientation(dI_dX, dI_dY, rows, cols, n_threads);
    rounded_grad_oreo = calculate_rounded_gradient(grad_oreo, roundval, rows, n_threads);
    return convert_matrixes_to_numpy_array({grad_oreo, rounded_grad_oreo});
}

py::array_t<double> EdgeDetector::get_non_max_suppresion()
{
    std::cout << "Calculate non maxima suppression\n";
    grad_mag2 = non_max_suppresion(rounded_grad_oreo, gradient_magnitued, n_threads);

    double max = grad_mag2[0][0];
    int index1 = 0, index2 = 0;
    for (int i = 0; i < grad_mag2.size(); i++)
    {
        for (int j = 0; j < grad_mag2[0].size(); j++)
        {
            if (grad_mag2[i][j] > max)
            {
                max = grad_mag2[i][j];
                index1 = i;
                index2 = j;
            }
        }
    }

    std::cout << "Max value: " << max << " at index: " << index1 << " " << index2 << std::endl;

    return convert_matrixes_to_numpy_array({grad_mag2})[0];
}

py::array_t<double> EdgeDetector::get_thresholded_img(double maxx, double minn, double meann)
{
    thresholded_img = non_max_threshold(grad_mag2, maxx, minn, meann, n_threads);

    return convert_matrixes_to_numpy_array({thresholded_img})[0];
}

py::array_t<double> EdgeDetector::get_hysteresis_img()
{
    hyst_img = hysteresis(thresholded_img);

    return convert_matrixes_to_numpy_array({hyst_img})[0];
}