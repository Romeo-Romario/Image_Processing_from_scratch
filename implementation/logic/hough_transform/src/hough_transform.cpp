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
    additional_functions::print_vec<double>(theta_angles, "theta_angles");
    rho_values = additional_functions::arange<double>(-diagonal, diagonal, rho);
    additional_functions::print_vec<double>(rho_values, "rho_values");

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