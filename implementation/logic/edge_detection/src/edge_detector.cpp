#include "../include/edge_detector.hpp"

CannyEdgeDetector::CannyEdgeDetector(const CannyEdgeDetector &el)
{
    grey_image = el.grey_image;
    kernel_matrix = el.kernel_matrix;
    convolved_image = el.convolved_image;
    dI_dX = el.dI_dX;
    dI_dY = el.dI_dY;
    gradient_magnitued = el.gradient_magnitued;
}
CannyEdgeDetector::CannyEdgeDetector(const vector<vector<double>> &input_image)
{
    grey_image = input_image;
    rows = grey_image.size();
    cols = grey_image[0].size();
}
CannyEdgeDetector::CannyEdgeDetector(const py::array_t<double> &input_image)
{
    grey_image = additional_modules::matrix_converter::convert_numpy_array_to_matrix(input_image);
    rows = grey_image.size();
    cols = grey_image[0].size();
}

void CannyEdgeDetector::convolve_image(double sigma, bool output)
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

    additional_modules::threading::split_to_threads(rows,
                                                    n_threads,
                                                    convolve_chunk,
                                                    std::ref(grey_image),
                                                    std::ref(kernel_matrix),
                                                    std::ref(convolved_image));
}

void CannyEdgeDetector::accumulate_image_gradients()
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

        additional_modules::threading::split_to_threads(rows, n_threads, convolve_chunk, std::ref(convolved_image),
                                                        std::ref(derivative_matrixes[matrix_index]),
                                                        std::ref(*matrix_ptrs[matrix_index]));
    }

    if (matrix_ptrs[2]->empty())
    {
        matrix_ptrs[2]->resize(rows, std::vector<double>(cols, 0.0));
    }

    additional_modules::threading::split_to_threads(rows, n_threads, chunk_gradient_magnitute,
                                                    std::ref(*matrix_ptrs[0]),
                                                    std::ref(*matrix_ptrs[1]),
                                                    std::ref(*matrix_ptrs[2]));
}

void CannyEdgeDetector::accumulate_image_gradient_orientation()
{
    double roundval = M_PI / 4.0;
    grad_oreo = calculate_gradient_orientation(dI_dX, dI_dY, rows, cols, n_threads);
    rounded_grad_oreo = calculate_rounded_gradient(grad_oreo, roundval, rows, n_threads);
}

void CannyEdgeDetector::accumulate_non_max_suppresion()
{
    grad_mag2 = non_max_suppresion(rounded_grad_oreo, gradient_magnitued, n_threads);
}

void CannyEdgeDetector::accumulate_thresholded_img(double maxx, double minn, double meann, double hight_treshold, double low_treshold)
{
    thresholded_img = non_max_threshold(grad_mag2, maxx, minn, meann, hight_treshold, low_treshold, n_threads);
}

py::array_t<double> CannyEdgeDetector::accumulate_hysteresis_img()
{
    hyst_img = hysteresis(thresholded_img);

    return additional_modules::matrix_converter::convert_matrixes_to_numpy_array({hyst_img})[0];
}

py::array_t<double> CannyEdgeDetector::get_canny_img(const py::array_t<double> &input_image,
                                                     double sigma,
                                                     double hight_threshold,
                                                     double low_threshold)
{
    // This method will carry all the logic till return of final Canny image

    grey_image = additional_modules::matrix_converter::convert_numpy_array_to_matrix(input_image);
    rows = grey_image.size();
    cols = grey_image[0].size();

    this->convolve_image(sigma);

    this->accumulate_image_gradients();

    this->accumulate_image_gradient_orientation();

    this->accumulate_non_max_suppresion();

    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();
    double sum = 0.0;
    size_t total_count = 0;

    for (const auto &row : grad_mag2)
    {
        if (row.empty())
            continue;

        auto result = std::minmax_element(row.begin(), row.end());

        min_val = std::min(min_val, *result.first);
        max_val = std::max(max_val, *result.second);

        sum += std::accumulate(row.begin(), row.end(), 0.0);

        total_count += row.size();
    }

    double mean_val = (total_count > 0) ? (sum / total_count) : 0.0;

    this->accumulate_thresholded_img(max_val, min_val, mean_val, hight_threshold, low_threshold);

    py::array_t<double> result = this->accumulate_hysteresis_img();

    return result;
}

py::array_t<double> CannyEdgeDetector::get_convolved_image()
{
    if (convolved_image.empty())
    {
        throw std::runtime_error("Convolved image not initialized. Call get_canny_img() first.");
    }
    return additional_modules::matrix_converter::convert_matrix_to_numpy_array(convolved_image);
}
vector<py::array_t<double>> CannyEdgeDetector::get_image_gradients()
{
    if (dI_dX.empty())
    {
        throw std::runtime_error("Gradient matrices not initialized. Call get_canny_img() first.");
    }
    return additional_modules::matrix_converter::convert_matrixes_to_numpy_array({dI_dX, dI_dY, gradient_magnitued});
}
vector<py::array_t<double>> CannyEdgeDetector::get_image_gradient_orientation()
{
    if (grad_oreo.empty())
    {
        throw std::runtime_error("Gradient orientation not initialized. Call get_canny_img() first.");
    }
    return additional_modules::matrix_converter::convert_matrixes_to_numpy_array({grad_oreo, rounded_grad_oreo});
}
py::array_t<double> CannyEdgeDetector::get_non_max_suppresion()
{
    if (grad_mag2.empty())
    {
        throw std::runtime_error("Non-max suppression matrix not initialized. Call get_canny_img() first.");
    }
    return additional_modules::matrix_converter::convert_matrix_to_numpy_array(grad_mag2);
}
py::array_t<double> CannyEdgeDetector::get_thresholded_img()
{
    if (thresholded_img.empty())
    {
        throw std::runtime_error("Thresholded image not initialized. Call get_canny_img() first.");
    }
    return additional_modules::matrix_converter::convert_matrix_to_numpy_array(thresholded_img);
}
py::array_t<double> CannyEdgeDetector::get_hysteresis_img()
{
    if (hyst_img.empty())
    {
        throw std::runtime_error("Hysteresis image not initialized. Call get_canny_img() first.");
    }
    return additional_modules::matrix_converter::convert_matrix_to_numpy_array(hyst_img);
}