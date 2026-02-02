#include "../include/hough_transform.hpp"
#include "../include/additional_functions.hpp"

using std::cout;
using std::endl;

HoughTransform::HoughTransform(const HoughTransform &el)
{
    edges = el.edges;
    theta = el.theta;
    rho = el.rho;
    threshold = el.threshold;
}
HoughTransform::HoughTransform(const py::array_t<double> &input_edges, const double theta, const double rho)
{
    edges = additional_modules::matrix_converter::convert_numpy_array_to_matrix(input_edges);
    this->theta = theta;
    this->rho = rho;
    this->threshold = 0.0;
}
vector<Matrix> HoughTransform::hough_lines(double threshold, double min_theta, double max_theta)
{
    diagonal = std::sqrt(edges.size() * edges.size() + edges[0].size() * edges[0].size());

    theta_angles = additional_functions::arange<double>(min_theta, max_theta, theta);
    // additional_functions::print_vec<double>(theta_angles, "theta_angles");
    rho_values = additional_functions::arange<double>(-diagonal, diagonal, rho);
    // additional_functions::print_vec<double>(rho_values, "rho_values");

    num_thetas = theta_angles.size();
    num_rhos = rho_values.size();

    cout << "Size of accumulator: " << num_rhos << " x " << num_thetas << endl;

    accumulator = Matrix(num_rhos, std::vector<double>(num_thetas));

    vector<double> sins(theta_angles.size(), 0), coss(theta_angles.size(), 0);

    for (int i = 0; i < theta_angles.size(); i++)
    {
        sins[i] = std::sin(theta_angles[i]);
        coss[i] = std::cos(theta_angles[i]);
    }

    // Finding edges indexes
    vector<std::pair<int, int>> edges_indexes;

    for (int i = 0; i < edges.size(); i++)
    {
        for (int j = 0; j < edges[0].size(); j++)
        {
            if (edges[i][j] > 1)
            {
                edges_indexes.emplace_back(i, j);
            }
        }
    }

    // Calculate parameter space plane
    double current_rho;
    int rho_pos;

    // Optimization idea
    double rho_min_val = -diagonal;
    for (int index = 0; index < edges_indexes.size(); index++)
    {
        for (int thetas_index = 0; thetas_index < theta_angles.size(); thetas_index++)
        {
            current_rho = edges_indexes[index].first * coss[thetas_index] + edges_indexes[index].second * sins[thetas_index];
            // rho_pos = additional_functions::rho_array_argmin(rho_values, current_rho);
            rho_pos = std::round((current_rho - rho_min_val) / rho);
            accumulator[rho_pos][thetas_index] += 1;
        }
    }

    // final <rho,theta> indexes

    vector<std::pair<int, int>> final_rho_theta_indexes;

    for (int i = 0; i < accumulator.size(); i++)
    {
        for (int j = 0; j < accumulator[0].size(); j++)
        {
            if (accumulator[i][j] > threshold)
            {
                final_rho_theta_indexes.emplace_back(i, j);
            }
        }
    }

    // For easier convertation to python I will avoid using std::pair at last step
    Matrix polar_coordinates = vector(final_rho_theta_indexes.size(), std::vector<double>(2, 0.0));

    for (int index = 0; index < final_rho_theta_indexes.size(); index++)
    {
        polar_coordinates[index][0] = rho_values[final_rho_theta_indexes[index].first];
        polar_coordinates[index][1] = theta_angles[final_rho_theta_indexes[index].second];
    }

    return {accumulator, polar_coordinates};
}

double HoughTransform::get_deskew_angle(double threshold, double min_theta, double max_theta)
{
    this->hough_lines(threshold, min_theta, max_theta);

    int max_votes = 0;
    int best_theta_index = 0;

    for (int r = 0; r < num_rhos; r++)
    {
        for (int t = 0; t < num_thetas; t++)
        {
            if (accumulator[r][t] > max_votes)
            {
                max_votes = accumulator[r][t];
                best_theta_index = t;
            }
        }
    }

    double best_theta_rad = theta_angles[best_theta_index];
    double best_theta_deg = best_theta_rad * (180.0 / 3.14159265358979323846);

    // 4. Calculate Rotation required to make this line Horizontal
    // Standard Hough: Horizontal line has normal at 90 degrees.
    double rotation_angle = 90 - best_theta_deg;

    return rotation_angle;
}

Matrix HoughTransform::get_rotation_matrix(std::pair<int, int> center, double angle, double scale)
{
    double angle_rad = angle * (M_PI / 180.0);

    double alpha = scale * std::cos(angle_rad);
    double betta = scale * std::sin(angle_rad);

    return {{alpha, betta, (1 - alpha) * center.first - betta * center.second},
            {-betta, alpha, betta * center.first + (1 - alpha) * center.second}};
}

py::array_t<double> HoughTransform::rotate_image(const Matrix &image, Matrix &rotation_matrix)
{
    int rows = image.size();
    int cols = image[0].size();

    Matrix deswed_image(rows, vector<double>(cols, 0.0));

    auto rotate_chunk = [](Matrix &deswed_image, const Matrix &image, const Matrix &rotation_matrix, int start_row, int end_row)
    {
        int rows = image.size();
        int cols = image[0].size();
        double x_t, y_t;

        double floor_x, floor_y, ceil_x, ceil_y;
        double w_top_left, w_top_right, w_bottom_left, w_bottom_right;
        double dx, dy;
        for (int row = start_row; row < end_row; row++)
        {
            for (int col = 0; col < deswed_image[0].size(); col++)
            {
                x_t = rotation_matrix[0][0] * row + rotation_matrix[0][1] * col + rotation_matrix[0][2];
                y_t = rotation_matrix[1][0] * row + rotation_matrix[1][1] * col + rotation_matrix[1][2];

                if (x_t < 0.0 || y_t < 0.0 || x_t >= (rows - 1) || y_t >= (cols - 1))
                {
                    continue;
                }

                floor_x = std::floor(x_t);
                floor_y = std::floor(y_t);
                ceil_x = std::ceil(x_t);
                ceil_y = std::ceil(y_t);

                dx = x_t - floor_x;
                dy = y_t - floor_y;

                w_top_left = (1.0 - dx) * (1.0 - dy);
                w_top_right = (1.0 - dx) * dy;
                w_bottom_left = dx * (1.0 - dy);
                w_bottom_right = dx * dy;

                deswed_image[row][col] = image[floor_x][floor_y] * w_top_left +
                                         image[floor_x][ceil_y] * w_top_right +
                                         image[ceil_x][floor_y] * w_bottom_left +
                                         image[ceil_x][ceil_y] * w_bottom_right;
            }
        }
    };

    additional_modules::threading::split_to_threads(rows, n_threads, rotate_chunk, std::ref(deswed_image), std::ref(image), std::ref(rotation_matrix));

    return additional_modules::matrix_converter::convert_matrix_to_numpy_array(deswed_image);
}

py::array_t<double> HoughTransform::deskew_image(const py::array_t<double> &image, double threshold, double min_theta, double max_theta)
{

    Matrix matrix_image = additional_modules::matrix_converter::convert_numpy_array_to_matrix(image);

    double rotation_angle = this->get_deskew_angle(threshold, min_theta, max_theta);

    cout << "Deskew Angle: " << rotation_angle << endl;

    std::pair<int, int> center(matrix_image[0].size() / 2, matrix_image.size() / 2);

    Matrix rotation_matrix = this->get_rotation_matrix(center, rotation_angle, 1.0);

    return this->rotate_image(matrix_image, rotation_matrix);
}