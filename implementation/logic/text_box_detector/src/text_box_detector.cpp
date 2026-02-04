#include "../include/text_box_detector.hpp"
#include "text_box_detector.hpp"

TextBoxDetector::TextBoxDetector(const py::array_t<double> &deskew_canny_image)
{
    this->deskew_canny_image = additional_modules::matrix_converter::convert_numpy_array_to_matrix(deskew_canny_image);

    cout << "Constructor worked correctly" << endl;
}

vector<double> TextBoxDetector::smooth_row_function()
{
    // First step get kernel 1D kernel

    vector<double> kernel = gaussian_kernel_1d(10.0);

    for (const auto &el : kernel)
    {
        cout << el << " ";
    }
    cout << endl;

    // Calculate vector of mean values by row
    int rows = deskew_canny_image.size();

    vector<double> one_dimentional_img_f = vector(rows, 0.0);
    auto calculate_mean_chunk = [](vector<double> &one_dimentional_img_f, const Matrix &deskew_canny_image, int start_row, int end_row)
    {
        int rows = deskew_canny_image.size();
        int cols = deskew_canny_image[0].size();
        double row_value;
        for (int i = start_row; i < end_row; i++)
        {
            row_value = 0.0;
            for (int j = 0; j < cols; j++)
            {
                row_value += deskew_canny_image[i][j];
            }

            one_dimentional_img_f[i] = (row_value / cols);
        }
    };

    additional_modules::threading::split_to_threads(rows, n_threads, calculate_mean_chunk, std::ref(one_dimentional_img_f), std::ref(deskew_canny_image));

    // Smooth the row signal
    vector<double> smoothed_img_f = vector(rows, 0.0);

    auto smooth_chunk = [](vector<double> &smoothed_img_f, const vector<double> one_dimentional_img_f, const vector<double> kernel, int start_row, int end_row)
    {
        int size = one_dimentional_img_f.size();
        int kernel_size = kernel.size();
        int kernel_half_size = kernel_size / 2;
        int kernel_index;

        for (int i = start_row; i < end_row; i++)
        {
            for (int k = 0; k < kernel_size; k++)
            {
                kernel_index = i - kernel_half_size + k;
                if (kernel_index < 0 || kernel_index > size - 1)
                    continue;
                smoothed_img_f[i] += one_dimentional_img_f[kernel_index] * kernel[k];
            }
        }
    };

    additional_modules::threading::split_to_threads(rows, n_threads, smooth_chunk, std::ref(smoothed_img_f), std::ref(one_dimentional_img_f), std::ref(kernel));

    this->smoothed_img_f = smoothed_img_f;
    return smoothed_img_f;
}
vector<bool> TextBoxDetector::find_extream_points()
{
    int size = smoothed_img_f.size();
    vector<bool> is_peak(size, false);

    double sum = std::accumulate(smoothed_img_f.begin(), smoothed_img_f.end(), 0.0);
    double global_average = sum / size;

    double value_threshold = global_average * 0.7;

    for (int i = 1; i < size - 1; i++)
    {
        double prev = smoothed_img_f[i - 1];
        double curr = smoothed_img_f[i];
        double next = smoothed_img_f[i + 1];

        if (curr < prev && curr < next)
        {
            if (curr < value_threshold)
            {
                is_peak[i] = true;
            }
        }
    }

    // 1. Calculate Average Distance (Logic corrected)
    double total_distance = 0.0;
    int number_of_segments = 0;

    // Find the very first peak to start
    int last_peak_pos = -1;
    for (int i = 0; i < size; i++)
    {
        if (is_peak[i])
        {
            last_peak_pos = i;
            break;
        }
    }

    // First Pass: Calculate Statistics
    for (int i = last_peak_pos + 1; i < size; i++)
    {
        if (is_peak[i])
        {
            int dist = i - last_peak_pos;
            total_distance += dist;
            number_of_segments++;
            last_peak_pos = i;
        }
    }

    // Handle case where no segments were found to avoid division by zero
    double mean_distance = total_distance / number_of_segments;
    double distance_threshold = mean_distance * 0.8;

    // 2. Second Pass: Filter (Logic corrected)
    // Reset tracker to the first peak again
    last_peak_pos = -1;
    for (int i = 0; i < size; i++)
    {
        if (is_peak[i])
        {
            last_peak_pos = i;
            break;
        }
    }

    cout << "Distance threshold: " << distance_threshold << endl;
    for (int i = last_peak_pos + 1; i < size; i++)
    {
        if (is_peak[i])
        {
            int current_distance = i - last_peak_pos;

            if (current_distance < distance_threshold)
            {
                // CONFLICT: Two peaks are too close.
                // STRATEGY: Delete the STRONGER one.

                if (smoothed_img_f[i] > smoothed_img_f[last_peak_pos])
                {
                    // Case A: Current peak (i) is stronger.
                    // Delete CURRENT.
                    is_peak[i] = false;

                    // 'last_peak_pos' stays the same (we keep the previous, weaker one).
                }
                else
                {
                    // Case B: Previous peak (last_peak_pos) was stronger.
                    // Delete PREVIOUS.
                    is_peak[last_peak_pos] = false;

                    // Update 'last_peak_pos' to the current one (we keep the current, weaker one).
                    last_peak_pos = i;
                }
            }
            else
            {
                // No conflict.
                // The current peak is valid (so far) and becomes the new reference.
                last_peak_pos = i;
            }
        }
    }

    return is_peak;
}