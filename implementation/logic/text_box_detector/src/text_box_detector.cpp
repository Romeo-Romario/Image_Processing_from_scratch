#include "../include/text_box_detector.hpp"
#include "text_box_detector.hpp"

TextBoxDetector::TextBoxDetector(const py::array_t<double> &deskew_canny_image)
{
    this->deskew_canny_image = additional_modules::matrix_converter::convert_numpy_array_to_matrix(deskew_canny_image);
}

vector<double> TextBoxDetector::smooth_row_function()
{
    vector<double> kernel = gaussian_kernel_1d(10.0);

    for (const auto &el : kernel)
    {
        cout << el << " ";
    }
    cout << endl;

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

    auto smooth_chunk = [](vector<double> &smoothed_img_f, const vector<double> &one_dimentional_img_f, const vector<double> &kernel, int start_row, int end_row)
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

vector<bool> TextBoxDetector::find_extream_points(double global_average_threshold, double mean_distance_threshold)
{
    int size = smoothed_img_f.size();
    vector<bool> is_peak(size, false);

    double sum = std::accumulate(smoothed_img_f.begin(), smoothed_img_f.end(), 0.0);
    double global_average = sum / size;

    double value_threshold = global_average * global_average_threshold;

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

    double total_distance = 0.0;
    int number_of_segments = 0;
    int last_peak_pos = -1;
    for (int i = 0; i < size; i++)
    {
        if (is_peak[i])
        {
            last_peak_pos = i;
            break;
        }
    }
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

    double mean_distance = total_distance / number_of_segments;
    double distance_threshold = mean_distance * mean_distance_threshold;

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
                if (smoothed_img_f[i] > smoothed_img_f[last_peak_pos])
                {
                    is_peak[i] = false;
                }
                else
                {
                    is_peak[last_peak_pos] = false;
                    last_peak_pos = i;
                }
            }
            else
            {
                last_peak_pos = i;
            }
        }
    }

    for (size_t i = 0; i < size; i++)
    {
        if (is_peak[i])
        {
            indexes_of_rows_extreame_points.push_back(i);
        }
    }

    return is_peak;
}

vector<py::array_t<double>> TextBoxDetector::get_text_rows()
{
    struct SegmentRange
    {
        int start_y;
        int end_y;
    };

    vector<SegmentRange> ranges;
    int rows = deskew_canny_image.size();

    int previous_index = 0;

    for (int split_idx : indexes_of_rows_extreame_points)
    {
        if (split_idx > previous_index && split_idx <= rows)
        {
            ranges.push_back({previous_index, split_idx});
            previous_index = split_idx;
        }
    }
    if (previous_index < rows)
    {
        ranges.push_back({previous_index, rows});
    }

    int num_segments = ranges.size();
    vector<Matrix> text_rows_matrices(num_segments);

    auto copy_chunk = [](
                          vector<Matrix> &dest_matrices,
                          const vector<SegmentRange> &ranges,
                          const Matrix &source_image,
                          int start_segment_idx,
                          int end_segment_idx)
    {
        int cols = source_image[0].size();

        for (int i = start_segment_idx; i < end_segment_idx; i++)
        {
            int y_start = ranges[i].start_y;
            int y_end = ranges[i].end_y;
            int height = y_end - y_start;

            Matrix segment(height, vector<double>(cols));

            for (int r = 0; r < height; r++)
            {
                segment[r] = source_image[y_start + r];
            }

            dest_matrices[i] = std::move(segment);
        }
    };

    additional_modules::threading::split_to_threads(
        num_segments,
        n_threads,
        copy_chunk,
        std::ref(text_rows_matrices),
        std::ref(ranges),
        std::ref(deskew_canny_image));

    vector<py::array_t<double>> python_result;
    python_result.reserve(num_segments);

    for (const auto &mat : text_rows_matrices)
    {
        python_result.push_back(
            additional_modules::matrix_converter::convert_matrix_to_numpy_array(mat));
    }

    text_rows = text_rows_matrices;

    return python_result;
}

std::pair<vector<vector<double>>, vector<vector<bool>>> TextBoxDetector::seperate_main_text()
{
    // FIRST PART of the function
    // Calculate sum as function for each row
    // Find extreame points as 0 that stays near non-zero values
    int cols = deskew_canny_image[0].size();
    int number_of_textrows = text_rows.size();

    vector<vector<double>> text_row_signal(number_of_textrows, vector<double>(cols, 0.0));
    vector<vector<bool>> near_zero_extrame_points(number_of_textrows, vector<bool>(cols, false));

    auto calculate_chunck_sum_and_non_zero = [](const Matrix &text_row, vector<vector<double>> &text_row_signal, int height, int current_index, int start_col, int end_col)
    {
        for (int r = 0; r < height; r++)
        {
            for (int c = start_col; c < end_col; c++)
            {
                text_row_signal[current_index][c] += text_row[r][c];
            }
        }
    };

    for (int i = 0; i < number_of_textrows; i++)
    {
        const auto &current_row_matrix = text_rows[i];
        int height = current_row_matrix.size();

        additional_modules::threading::split_to_threads(cols, n_threads, calculate_chunck_sum_and_non_zero,
                                                        std::ref(current_row_matrix), std::ref(text_row_signal), height, i);

        for (int col = 1; col < cols - 1; col++)
        {
            if (text_row_signal[i][col] == 0 && (text_row_signal[i][col - 1] > 0 || text_row_signal[i][col + 1] > 0))
                near_zero_extrame_points[i][col] = true;
        }
    }

    // SECOND PART
    // Convolve signal

    vector<double> kernel = gaussian_kernel_1d(10.0); // BIG CONVOLUTION IS ESSENTIAL
    vector<vector<double>> convolved_text_row_signal(number_of_textrows, vector<double>(cols, 0.0));

    auto smooth_chunk = [](vector<double> &smoothed_img_f, const vector<double> one_dimentional_img_f, const vector<double> &kernel, int start_row, int end_row)
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

    for (int text_row_index = 0; text_row_index < number_of_textrows; text_row_index++)
    {
        additional_modules::threading::split_to_threads(cols, n_threads, smooth_chunk,
                                                        std::ref(convolved_text_row_signal[text_row_index]),
                                                        std::ref(text_row_signal[text_row_index]), std::ref(kernel));
    }

    // Go through extreame points and find the longest uninterapted part of function

    for (int text_row_index = 0; text_row_index < number_of_textrows; text_row_index++)
    {
        bool start = false;
        int start_index = 0;
        int end_index = 0;

        int max_start_index = 0;
        int max_end_index = 0;
        int max_length = -1;

        const auto &current_row_signal = convolved_text_row_signal[text_row_index];
        auto &current_row_extreame = near_zero_extrame_points[text_row_index];
        for (int i = 0; i < cols; i++)
        {
            if (current_row_extreame[i] && !start)
            {
                start = true;
                start_index = i;
            }
            else if (start && current_row_signal[i] > 2.0 && current_row_extreame[i])
            {
                current_row_extreame[i] = false;
                end_index = i;
            }
            else if (current_row_signal[i] < 2.0 && start)
            {
                start = false;

                if (max_length < (end_index - start_index))
                {
                    max_start_index = start_index;
                    max_end_index = end_index;
                    max_length = (end_index - start_index);
                }

                current_row_extreame[end_index] = true;
            }
        }

        current_row_extreame = vector<bool>(cols, false);
        current_row_extreame[max_start_index] = true;
        current_row_extreame[max_end_index] = true;
    }

    return {convolved_text_row_signal, near_zero_extrame_points};
}
