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
    text_rows.resize(num_segments);

    auto copy_chunk = [](
                          vector<TextRow> &dest_matrices,
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

            dest_matrices[i].text_matrix = std::move(segment);
            dest_matrices[i].y_start = y_start;
            dest_matrices[i].y_end = y_end;
        }
    };

    additional_modules::threading::split_to_threads(
        num_segments,
        n_threads,
        copy_chunk,
        std::ref(text_rows),
        std::ref(ranges),
        std::ref(deskew_canny_image));

    vector<py::array_t<double>> python_result;
    python_result.reserve(num_segments);

    int index = 0;
    for (auto &mat : text_rows)
    {
        mat.index_in_original_img = index;
        python_result.push_back(
            additional_modules::matrix_converter::convert_matrix_to_numpy_array(mat.text_matrix));
        index++;
    }

    return python_result;
}

std::pair<vector<vector<double>>, vector<vector<bool>>> TextBoxDetector::seperate_main_text()
{
    // separate text in a row

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
        const auto &current_row_matrix = text_rows[i].text_matrix;
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

    vector<std::pair<int, int>> main_text_indexes;
    main_text_indexes.resize(number_of_textrows);

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

        std::fill(current_row_extreame.begin(), current_row_extreame.end(), false);
        current_row_extreame[max_start_index] = true;
        current_row_extreame[max_end_index] = true;

        main_text_indexes[text_row_index] = {max_start_index, max_end_index};
    }

    // Apply boundaries to each text row matrix -> Clean text

    for (int text_row_index = 0; text_row_index < number_of_textrows; text_row_index++)
    {
        const auto &text_row = text_rows[text_row_index].text_matrix;
        int col_start = main_text_indexes[text_row_index].first;
        int col_end = main_text_indexes[text_row_index].second;

        text_rows[text_row_index].x_start = col_start;
        text_rows[text_row_index].x_end = col_end;

        int size_clean_text_columns = col_end - col_start + 1;
        int rows = text_row.size();

        Matrix clean_text = vector(rows, vector(size_clean_text_columns, 0.0));
        for (int row = 0; row < rows; row++)
        {
            for (int i = 0; i < size_clean_text_columns; i++)
            {
                clean_text[row][i] = text_row[row][col_start + i];
            }
        }

        text_rows[text_row_index].text_matrix = clean_text;
    }

    return {convolved_text_row_signal, near_zero_extrame_points};
}

vector<py::array_t<double>> TextBoxDetector::get_clean_text_rows()
{
    vector<py::array_t<double>> result;
    result.reserve(text_rows.size());
    int index = 0;
    for (const auto &el : text_rows)
    {
        result.push_back(additional_modules::matrix_converter::convert_matrix_to_numpy_array(el.text_matrix));

        cout << "TextRow numer: " << index << endl;
        cout << "Matrix size: " << el.text_matrix.size() << " " << el.text_matrix[0].size() << endl;
        cout << "Start Y: " << el.y_start << " End Y: " << el.y_end << endl;
        cout << "Start X: " << el.x_start << " End X: " << el.x_end << endl
             << endl;

        index++;
    }

    return result;
}

void TextBoxDetector::remove_rows_without_text(double density_threshold, int width_threshold)
{
    double density;
    vector<bool> should_be_removed(text_rows.size(), false);
    int remove_number = 0;
    int index = 0;
    for (const auto &el : text_rows)
    {
        if (el.x_end - el.x_start < width_threshold)
        {
            should_be_removed[index] = true;
            remove_number++;
            index++;
            if (el.x_end - el.x_start < width_threshold)
                cout << "Remove text row " << index << " cause it's width: " << el.x_end - el.x_start << " < " << width_threshold << endl;
            continue;
        }

        const auto &matrix = el.text_matrix;
        double pixels_sum = 0.0;
        for (int i = 0; i < matrix.size(); i++)
        {
            for (int j = 0; j < matrix[0].size(); j++)
            {
                if (matrix[i][j] > 0.1)
                {
                    pixels_sum += matrix[i][j];
                }
            }
        }

        density = pixels_sum / (matrix.size() * matrix[0].size());
        if (density < density_threshold)
        {
            cout << "Remove text row " << index << " cause density: " << density << " < " << density_threshold << endl;
            should_be_removed[index] = true;
            remove_number++;
        }
        index++;
    }

    vector<TextRow> cleaned_text_rows(text_rows.size() - remove_number);
    int push_index = 0; // TODO: rename
    for (int i = 0; i < should_be_removed.size(); i++)
    {
        if (should_be_removed[i])
            continue;
        cleaned_text_rows[push_index] = text_rows[i];
        push_index++;
    }

    text_rows = cleaned_text_rows;
    cout << "Text rows size after cleaning: " << text_rows.size() << endl;
}

vector<TextRow> TextBoxDetector::detect_symbol_boxes(float pixel_threshold)
{
    // calculate 1d function for each text row
    auto calculate_1d_func = [](vector<TextRow> &text_rows, int start, int end)
    {
        for (int index = start; index < end; index++)
        {
            const auto &matrix = text_rows[index].text_matrix;

            int rows = matrix.size();
            int cols = matrix[0].size();

            auto &_1d_func = text_rows[index]._1d_function;

            _1d_func.assign(cols, 0.0);

            for (int col = 0; col < cols; col++)
            {
                for (int row = 0; row < rows; row++)
                {
                    _1d_func[col] += matrix[row][col];
                }
            }
        }
    };

    additional_modules::threading::split_to_threads(text_rows.size(), n_threads, calculate_1d_func, std::ref(text_rows));

    this->zero_division(pixel_threshold);

    return text_rows;
}

// void TextBoxDetector::zero_division(float pixel_threshold)
// {
//     double division_threshold = pixel_threshold * 256;
//     for (auto &el : text_rows)
//     {
//         auto &_1d_func = el._1d_function;
//         auto &zero_sep_points = el.zero_sep_points;
//         auto &potetional_zero_sep_points = el.potetional_zero_sep_points;
//         // we skip first 5 pixels and last 5 pixels to avoid additional zero division
//         for (int index = 5; index < _1d_func.size() - 5; index++)
//         {
//             // Check when zero is starting
//             if (_1d_func[index] < 5.0 && (_1d_func[index - 1] > 256.0 || _1d_func[index + 1] > 256.0))
//             {
//                 zero_sep_points.push_back(index);
//             }
//             if (_1d_func[index] > 256.0 && _1d_func[index] < division_threshold && (_1d_func[index - 1] > division_threshold + 1 || _1d_func[index + 1] > division_threshold + 1))
//             {
//                 potetional_zero_sep_points.push_back(index);
//             }
//         }
//     }
// }

void TextBoxDetector::zero_division(float pixel_threshold)
{
    double division_threshold = pixel_threshold * 256.0;
    for (auto &el : text_rows)
    {
        auto &_1d_func = el._1d_function;
        auto &zero_sep_points = el.zero_sep_points;
        auto &potetional_zero_sep_points = el.potetional_zero_sep_points;

        int func_size = static_cast<int>(_1d_func.size());
        for (int index = 5; index < func_size - 5; index++)
        {
            if (_1d_func[index] < 5.0 && (_1d_func[index - 1] > 256.0 || _1d_func[index + 1] > 256.0))
            {
                zero_sep_points.push_back(index);
                continue;
            }
            if (_1d_func[index] > 254.0 && _1d_func[index] < division_threshold && (_1d_func[index - 1] > division_threshold + 1.0 || _1d_func[index + 1] > division_threshold + 1.0))
            {
                potetional_zero_sep_points.push_back(index);
            }
        }
    }
}