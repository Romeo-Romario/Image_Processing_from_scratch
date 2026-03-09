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
    vector<int> is_peak(size, 0);

    double sum = std::accumulate(smoothed_img_f.begin(), smoothed_img_f.end(), 0.0);
    double global_average = sum / size;
    double value_threshold = global_average * global_average_threshold;

    // 1. Robust Plateau-Aware Minimum Detection
    for (int i = 1; i < size - 1; i++)
    {
        double curr = smoothed_img_f[i];

        // We only care about valleys that drop below the red threshold line
        if (curr < value_threshold)
        {
            // Check if we are at the bottom of a downward slope (signal stopped decreasing)
            if (curr <= smoothed_img_f[i - 1])
            {
                int j = i;

                // Scan forward to find the end of this flat valley bottom.
                // We use 1e-5 to safely compare floating point numbers that are "equal"
                while (j < size - 1 && std::abs(smoothed_img_f[j] - curr) < 1e-5)
                {
                    j++;
                }

                // If the signal goes UP after this flat area, it is a guaranteed valley!
                if (smoothed_img_f[j] > curr + 1e-5)
                {
                    // Place the cut line exactly in the center of the plateau
                    int plateau_center = i + (j - i) / 2;
                    is_peak[plateau_center] = 1;

                    // Fast-forward the loop past this plateau so we don't detect it twice
                    i = j - 1;
                }
            }
        }
    }

    // 2. Peaks filtering logic based on distance
    double total_distance = 0.0;
    int number_of_segments = 0;
    int last_peak_pos = -1;

    for (int i = 0; i < size; i++)
    {
        if (is_peak[i] == 1)
        {
            last_peak_pos = i;
            break;
        }
    }

    for (int i = last_peak_pos + 1; i < size; i++)
    {
        if (is_peak[i] == 1)
        {
            int dist = i - last_peak_pos;
            total_distance += dist;
            number_of_segments++;
            last_peak_pos = i;
        }
    }

    if (number_of_segments > 0)
    {
        double mean_distance = total_distance / number_of_segments;
        double distance_threshold = mean_distance * mean_distance_threshold;

        last_peak_pos = -1;
        for (int i = 0; i < size; i++)
        {
            if (is_peak[i] == 1)
            {
                last_peak_pos = i;
                break;
            }
        }

        for (int i = last_peak_pos + 1; i < size; i++)
        {
            if (is_peak[i] == 1)
            {
                int current_distance = i - last_peak_pos;
                if (current_distance < distance_threshold)
                {
                    if (smoothed_img_f[i] > smoothed_img_f[last_peak_pos])
                    {
                        is_peak[i] = 0;
                    }
                    else
                    {
                        is_peak[last_peak_pos] = 0;
                        last_peak_pos = i;
                    }
                }
                else
                {
                    last_peak_pos = i;
                }
            }
        }
    }

    // 3. Update the stored indexes property
    indexes_of_rows_extreame_points.clear();
    for (size_t i = 0; i < size; i++)
    {
        if (is_peak[i] == 1)
        {
            indexes_of_rows_extreame_points.push_back(i);
        }
    }

    // 4. Final conversion to boolean array
    vector<bool> result(size, false);
    for (int i = 0; i < size; i++)
    {
        if (is_peak[i] == 1)
        {
            result[i] = true;
        }
    }

    return result;
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
    this->refine_symbol_boundaries();
    this->normalize_symbol_boxes();

    return text_rows;
}

void TextBoxDetector::zero_division(float pixel_threshold)
{
    double division_threshold = pixel_threshold * 256.0;

    // DEBUGG
    int text_row_index = 0;

    for (auto &el : text_rows)
    {
        auto &_1d_func = el._1d_function;
        auto &zero_sep_points = el.zero_sep_points;
        auto &potetional_zero_sep_points = el.potetional_zero_sep_points;

        zero_sep_points = {1};

        int func_size = static_cast<int>(_1d_func.size());
        // We need to start/end with padding of 5 pixels to avoid unneccessary zero crossing logic
        for (int index = 5; index < func_size - 5; index++)
        {
            // If value of current column is 0 than we check left and right columns
            // non - zero -> separation point
            // zero -> we are in a middle of zero interval
            if (_1d_func[index] < 5.0 && (_1d_func[index - 1] > 5.0 || _1d_func[index + 1] > 5.0))
            {
                zero_sep_points.push_back(index);
                continue;
            }

            // There is a chance that symbols are connected with "noise" in that case we can use thresholding
            // To detect boundaries -> TODO: use mean value to validate if additional seperation is suitable
            if (_1d_func[index] > 5.0 && _1d_func[index] < division_threshold && (_1d_func[index - 1] > division_threshold || _1d_func[index + 1] > division_threshold))
            {
                potetional_zero_sep_points.push_back(index);
            }
        }

        zero_sep_points.push_back(_1d_func.size() - 1);
    }
}

void TextBoxDetector::refine_symbol_boundaries()
{
    // A multiplier to define what "too large" means.
    // 1.7 means 70% wider than the average symbol.
    const double TOO_LARGE_MULTIPLIER = 1.7;

    for (auto &row : text_rows)
    {
        const auto &zero_pts = row.zero_sep_points;
        const auto &pot_pts = row.potetional_zero_sep_points;
        auto &symbols_limits = row.symbols_limits;

        if (zero_pts.size() < 2)
            continue;

        // --- STEP 0: Extract only the segments that are actually text, not spaces ---
        struct Segment
        {
            int start_x;
            int end_x;
        };
        vector<Segment> text_segments;

        for (size_t i = 0; i < zero_pts.size() - 1; ++i)
        {
            int start_x = zero_pts[i];
            int end_x = zero_pts[i + 1];

            // Check the middle of the segment to see if it's text or empty space
            int mid_x = start_x + (end_x - start_x) / 2;
            if (row._1d_function[mid_x] >= 5.0)
            {
                text_segments.push_back({start_x, end_x});
            }
        }

        if (text_segments.empty())
            continue;

        // --- STEP 1: Find Mean Symbol Width using actual text segments ---
        double mean_width = -1.0;
        int chain_size = 5;
        bool force_dfs_for_row = false; // Add a flag to track short rows
        if (text_segments.size() >= chain_size)
        {
            double min_variance = 1e9;
            for (size_t i = 0; i <= text_segments.size() - chain_size; ++i)
            {
                double sum = 0.0;
                vector<double> widths(chain_size);

                for (int j = 0; j < chain_size; ++j)
                {
                    widths[j] = text_segments[i + j].end_x - text_segments[i + j].start_x;
                    sum += widths[j];
                }

                double local_mean = sum / chain_size;
                double variance = 0.0;
                for (double w : widths)
                {
                    variance += (w - local_mean) * (w - local_mean);
                }

                if (variance < min_variance)
                {
                    min_variance = variance;
                    mean_width = local_mean;
                }
            }
        }
        else
        {
            // FALLBACK FOR SHORT ROWS
            // If text rows have fewer than 5 symbols, they could potentially be
            // badly separated titles. Force DFS directly!
            force_dfs_for_row = true;
        }

        // --- STEP 2 & 3: Evaluate Divisions and Apply Thresholds/DFS ---

        if (force_dfs_for_row)
        {
            // BYPASS threshold logic entirely. Feed all segments directly to DFS.
            for (const auto &seg : text_segments)
            {
                this->extract_symbols_with_dfs(row, seg.start_x, seg.end_x);
            }
        }
        else
        {
            // Normal threshold logic for standard long rows
            for (const auto &seg : text_segments)
            {
                int start_x = seg.start_x;
                int end_x = seg.end_x;
                double width = end_x - start_x;

                if (width <= mean_width * TOO_LARGE_MULTIPLIER)
                {
                    // Good division
                    symbols_limits.push_back({{start_x + row.x_start, row.y_start}, {end_x + row.x_start, row.y_end}});
                }
                else
                {
                    vector<int> valid_potentials;
                    for (int p : pot_pts)
                    {
                        if (p > start_x && p < end_x)
                        {
                            valid_potentials.push_back(p);
                        }
                    }

                    bool resolved_with_thresholds = false;

                    if (!valid_potentials.empty())
                    {
                        int current_start = start_x;
                        bool still_too_large = false;

                        for (int vp : valid_potentials)
                        {
                            if ((vp - current_start) > mean_width * TOO_LARGE_MULTIPLIER)
                            {
                                still_too_large = true;
                                break;
                            }
                            current_start = vp;
                        }
                        if ((end_x - current_start) > mean_width * TOO_LARGE_MULTIPLIER)
                        {
                            still_too_large = true;
                        }

                        if (!still_too_large)
                        {
                            int temp_start = start_x;
                            for (int vp : valid_potentials)
                            {
                                symbols_limits.push_back({{temp_start + row.x_start, row.y_start}, {vp + row.x_start, row.y_end}});
                                temp_start = vp;
                            }
                            symbols_limits.push_back({{temp_start + row.x_start, row.y_start}, {end_x + row.x_start, row.y_end}});
                            resolved_with_thresholds = true;
                        }
                    }

                    if (!resolved_with_thresholds)
                    {
                        // Fallback to tracing the actual pixels
                        this->extract_symbols_with_dfs(row, start_x, end_x);
                    }
                }
            }
        }

        // --- STEP 4: Box Absorption (Merging Overlaps & Vertical Stacks) ---
        bool merged_any = true;
        while (merged_any && symbols_limits.size() > 1)
        {
            merged_any = false;
            for (size_t i = 0; i < symbols_limits.size(); ++i)
            {
                for (size_t j = i + 1; j < symbols_limits.size(); ++j)
                {
                    auto &b1 = symbols_limits[i];
                    auto &b2 = symbols_limits[j];

                    // 1. Standard overlap check (AABB collision)
                    bool x_overlap = (b1.first.x <= b2.second.x && b1.second.x >= b2.first.x);
                    bool y_overlap = (b1.first.y <= b2.second.y && b1.second.y >= b2.first.y);
                    bool aabb_collision = x_overlap && y_overlap;

                    // 2. Vertical Stack Check (Fixes 'i', 'ї', '!', '?', '=', ':')
                    // DFS separates disconnected components. We merge them if they sit
                    // directly above/below each other by checking their X-axis overlap.
                    bool vertical_stack = false;
                    if (x_overlap)
                    {
                        // Calculate how much they overlap on the X axis
                        int overlap_width = std::min(b1.second.x, b2.second.x) - std::max(b1.first.x, b2.first.x);

                        // Get the widths of both components
                        int w1 = b1.second.x - b1.first.x;
                        int w2 = b2.second.x - b2.first.x;
                        int min_w = std::min(w1, w2);

                        // If the overlap is substantial (e.g., > 40% of the smaller component's width)
                        // they belong to the same fragmented character.
                        if (min_w > 0 && ((double)overlap_width / min_w > 0.4))
                        {
                            vertical_stack = true;
                        }
                    }

                    // 3. Containment Check (one box is fully inside another)
                    bool b2_in_b1 = (b2.first.x >= b1.first.x && b2.second.x <= b1.second.x &&
                                     b2.first.y >= b1.first.y && b2.second.y <= b1.second.y);
                    bool b1_in_b2 = (b1.first.x >= b2.first.x && b1.second.x <= b2.second.x &&
                                     b1.first.y >= b2.first.y && b1.second.y <= b2.second.y);

                    // If ANY of these conditions are met, merge the boxes!
                    if (aabb_collision || vertical_stack || b2_in_b1 || b1_in_b2)
                    {
                        // Expand b1 to absorb b2
                        b1.first.x = std::min(b1.first.x, b2.first.x);
                        b1.first.y = std::min(b1.first.y, b2.first.y);
                        b1.second.x = std::max(b1.second.x, b2.second.x);
                        b1.second.y = std::max(b1.second.y, b2.second.y);

                        // Remove b2
                        symbols_limits.erase(symbols_limits.begin() + j);
                        merged_any = true;

                        // Break inner loop to restart with new geometry
                        break;
                    }
                }
                // Break outer loop to restart
                if (merged_any)
                    break;
            }
        }
    }
}

void TextBoxDetector::extract_symbols_with_dfs(TextRow &row, int start_x, int end_x)
{
    int height = row.text_matrix.size();
    int local_width = end_x - start_x;

    // Visited grid for this specific segment only (saves memory and time)
    vector<vector<bool>> visited(height, vector<bool>(local_width, false));

    // 8-way connectivity directions
    int dx[] = {0, 0, -1, 1, -1, 1, -1, 1};
    int dy[] = {-1, 1, 0, 0, -1, -1, 1, 1};

    for (int y = 0; y < height; ++y)
    {
        for (int x = start_x; x < end_x; ++x)
        {
            int local_x = x - start_x;

            // If unvisited and pixel is text (thresholded > 0.1 to avoid faint noise)
            if (!visited[y][local_x] && row.text_matrix[y][x] > 0.1)
            {
                // Start DFS for a new connected symbol
                int sym_min_x = x;
                int sym_max_x = x;
                int sym_min_y = y;
                int sym_max_y = y;

                std::stack<Point> pixel_stack;
                pixel_stack.push({x, y});
                visited[y][local_x] = true;

                while (!pixel_stack.empty())
                {
                    Point curr = pixel_stack.top();
                    pixel_stack.pop();

                    // Expand the bounding box as we discover the symbol
                    if (curr.x < sym_min_x)
                        sym_min_x = curr.x;
                    if (curr.x > sym_max_x)
                        sym_max_x = curr.x;
                    if (curr.y < sym_min_y)
                        sym_min_y = curr.y;
                    if (curr.y > sym_max_y)
                        sym_max_y = curr.y;

                    // Check all 8 neighbors
                    for (int i = 0; i < 8; ++i)
                    {
                        int nx = curr.x + dx[i];
                        int ny = curr.y + dy[i];

                        // Stay within the boundaries of our specific oversized segment
                        if (nx >= start_x && nx < end_x && ny >= 0 && ny < height)
                        {
                            int n_local_x = nx - start_x;
                            if (!visited[ny][n_local_x] && row.text_matrix[ny][nx] > 0.1)
                            {
                                visited[ny][n_local_x] = true;
                                pixel_stack.push({nx, ny});
                            }
                        }
                    }
                }

                // Filter out tiny noise (e.g., 1x1 or 2x2 isolated pixel blocks)
                if ((sym_max_x - sym_min_x > 2) && (sym_max_y - sym_min_y > 2))
                {
                    // Translate local X and local Y back to the absolute coordinates
                    Point top_left = {sym_min_x + row.x_start, row.y_start + sym_min_y};
                    Point bottom_right = {sym_max_x + row.x_start, row.y_start + sym_max_y};

                    row.symbols_limits.push_back({top_left, bottom_right});
                }
            }
        }
    }
}

void TextBoxDetector::normalize_symbol_boxes()
{
    for (auto &row : text_rows)
    {
        auto &symbols_limits = row.symbols_limits;
        const auto &matrix = row.text_matrix;

        if (matrix.empty())
            continue;

        int height = matrix.size();
        int width = matrix[0].size();

        // We will build a new vector to keep only valid, non-empty boxes
        vector<std::pair<Point, Point>> normalized_limits;

        for (auto &box : symbols_limits)
        {
            // Translate global coordinates back to local text_matrix coordinates
            int local_start_x = box.first.x - row.x_start;
            int local_end_x = box.second.x - row.x_start;
            int local_start_y = box.first.y - row.y_start;
            int local_end_y = box.second.y - row.y_start;

            // Safety bounds clamp
            local_start_x = std::max(0, local_start_x);
            local_end_x = std::min(width, local_end_x);
            local_start_y = std::max(0, local_start_y);
            local_end_y = std::min(height, local_end_y);

            int true_top = -1;
            int true_bottom = -1;
            int true_left = -1;
            int true_right = -1;

            // 1. Find True Top
            for (int y = local_start_y; y < local_end_y; ++y)
            {
                for (int x = local_start_x; x < local_end_x; ++x)
                {
                    if (matrix[y][x] > 0.1)
                    {
                        true_top = y;
                        break;
                    }
                }
                if (true_top != -1)
                    break;
            }

            // If true_top is still -1, the box is completely empty (just noise). Skip it!
            if (true_top == -1)
                continue;

            // 2. Find True Bottom
            for (int y = local_end_y - 1; y >= local_start_y; --y)
            {
                for (int x = local_start_x; x < local_end_x; ++x)
                {
                    if (matrix[y][x] > 0.1)
                    {
                        true_bottom = y;
                        break;
                    }
                }
                if (true_bottom != -1)
                    break;
            }

            // 3. Find True Left
            for (int x = local_start_x; x < local_end_x; ++x)
            {
                for (int y = true_top; y <= true_bottom; ++y)
                { // Only search between true top/bottom
                    if (matrix[y][x] > 0.1)
                    {
                        true_left = x;
                        break;
                    }
                }
                if (true_left != -1)
                    break;
            }

            // 4. Find True Right
            for (int x = local_end_x - 1; x >= local_start_x; --x)
            {
                for (int y = true_top; y <= true_bottom; ++y)
                {
                    if (matrix[y][x] > 0.1)
                    {
                        true_right = x;
                        break;
                    }
                }
                if (true_right != -1)
                    break;
            }

            // Convert back to global coordinates and save
            normalized_limits.push_back({{true_left + row.x_start, true_top + row.y_start},
                                         {true_right + row.x_start + 1, true_bottom + row.y_start + 1}});
        }

        // Overwrite the old, loose boxes with the new, tight boxes
        row.symbols_limits = normalized_limits;
    }
}