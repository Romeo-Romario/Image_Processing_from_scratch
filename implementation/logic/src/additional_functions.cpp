#include "../include/additional_functions.hpp"
#include "../include/threading.hpp"
using std::signbit;

Matrix gaussian_kernel(const double sigma)
{
    // Compute kernel size (ensure it's odd)
    int size = static_cast<int>(2 * std::ceil(3 * sigma) + 1);
    int half = size / 2;

    Matrix kernel(size, std::vector<double>(size));
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
    const Matrix &grey_image,
    const Matrix &kernel,
    Matrix &result,
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

Matrix calculate_gradient_orientation(const Matrix &dI_dX, const Matrix &dI_dY, int rows, int cols, int n_threads)
{

    Matrix grad_oreo(rows, vector<double>(cols, 0.0001));

    auto chunk_gradient_orientation = [](Matrix &grad_oreo, const Matrix &dI_dX, const Matrix &dI_dY, int start_row, int end_row)
    {
        for (int row = start_row; row < end_row; row++)
        {
            for (int col = 0; col < grad_oreo[0].size(); col++)
            {
                grad_oreo[row][col] = std::atan2(dI_dY[row][col], (dI_dX[row][col] + grad_oreo[row][col]));
            }
        }
    };

    threading::split_to_threads(rows, n_threads, chunk_gradient_orientation, std::ref(grad_oreo), std::ref(dI_dX), std::ref(dI_dY));
    return grad_oreo;
}

Matrix calculate_rounded_gradient(const Matrix &grad_oreo, double roundval, int rows, int n_threads)
{
    Matrix rounded_grad_oreo(grad_oreo);

    auto chunk_rounded_gradient = [](Matrix &rounded_grad_oreo, double roundval, int start_row, int end_row)
    {
        double _1, _2, _3, _4, _5, _6;

        for (int row = start_row; row < end_row; row++)
        {
            for (int col = 0; col < rounded_grad_oreo[0].size(); col++)
            {
                rounded_grad_oreo[row][col] = std::ceil((std::floor(rounded_grad_oreo[row][col] / (roundval * 0.5))) / 2.0) * roundval;
            }
        }
    };

    threading::split_to_threads(rows, n_threads, chunk_rounded_gradient, std::ref(rounded_grad_oreo), roundval);
    return rounded_grad_oreo;
}

inline std::pair<int, int> get_direction_index(double angle)
{
    if (angle < 0)
        angle += 2 * M_PI;

    int sector = static_cast<int>(std::round(angle / (M_PI / 4.0))) % 8;

    return direction_offsets[sector];
}

Matrix non_max_suppresion(const Matrix &rounded_grad_oreo, const Matrix &gradient_magnitued, int n_threads)
{
    auto suppresed_matrix(gradient_magnitued);
    int cols = suppresed_matrix[0].size();
    // Make frame to 0.0
    std::fill(suppresed_matrix[0].begin(), suppresed_matrix[0].end(), 0.0);
    for (size_t row = 0; row < suppresed_matrix.size(); row++)
    {
        suppresed_matrix[row][0] = 0.0;
        suppresed_matrix[row][cols - 1] = 0.0;
    }

    std::pair<double, double> _1_neighbor_pixel, _2_neighbor_pixel;
    auto chunk_suppresion = [](Matrix &suppresed_matrix,
                               const Matrix &gradient_magnitued,
                               const Matrix &rounded_grad_oreo,
                               int start_row, int end_row)
    {
        if (start_row == 0)
            start_row++;
        if (end_row == suppresed_matrix.size())
            end_row--;
        std::pair<int, int> shift;

        for (int row = start_row; row < end_row; row++)
        {
            for (int col = 1; col < suppresed_matrix[0].size() - 1; col++)
            {
                shift = get_direction_index(rounded_grad_oreo[row][col]);

                int r1 = row + shift.first;
                int c1 = col + shift.second;

                int r2 = row - shift.first;
                int c2 = col - shift.second;

                double current_val = gradient_magnitued[row][col];
                double val1 = gradient_magnitued[r1][c1];
                double val2 = gradient_magnitued[r2][c2];

                if (current_val < val1 || current_val < val2)
                {
                    suppresed_matrix[row][col] = 0.0;
                }
            }
        }
    };
    threading::split_to_threads(rounded_grad_oreo.size(), n_threads, chunk_suppresion, std::ref(suppresed_matrix),
                                std::ref(gradient_magnitued), std::ref(rounded_grad_oreo));
    return suppresed_matrix;
}

Matrix non_max_threshold(const Matrix &non_max_suppr_img, double maxx, double minn, double meann, int n_threads)
{
    using std::cout;
    using std::endl;

    auto img(non_max_suppr_img);

    cout << "Image shape: " << img.size() << img[0].size();

    double high_threshold_multiplier = 0.25;
    double h = maxx * high_threshold_multiplier;

    double low_threshold_multiplier = 0.06;
    double l = h * low_threshold_multiplier;

    cout << "\nValues:\n"
         << "max: " << maxx << " min: " << minn << "  mean: " << meann << endl;
    cout << "h: " << h << " l: " << l << endl;

    auto chunk_thresholding = [](Matrix &img, double l, double h, double maxx, double mean, int start_row, int end_row)
    {
        int condition;
        for (int row = start_row; row < end_row; row++)
        {
            for (int col = 0; col < img[0].size(); col++)
            {
                if (img[row][col] < l)
                {
                    condition = 1;
                    img[row][col] = 0.0;
                }
                else if (img[row][col] >= l && img[row][col] <= h)
                {
                    condition = 2;
                    img[row][col] = mean;
                }
                else
                {
                    condition = 3;
                    img[row][col] = maxx;
                }
                // if (row > 116 && row < 130 && col > 297 && col < 315)
                // {
                //     cout << "Pixel at index: " << row << " " << col << " val:" << img[row][col] << " cond: " << condition << endl;
                // }
            }
        }
    };

    threading::split_to_threads(img.size(), n_threads, chunk_thresholding, std::ref(img), l, h, maxx, meann);

    return img;
}