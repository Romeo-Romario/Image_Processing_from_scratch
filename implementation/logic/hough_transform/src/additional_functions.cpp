#include "../include/additional_functions.hpp"
#include "additional_functions.hpp"

namespace additional_functions
{
    int rho_array_argmin(vector<double> rho_values, double current_rho)
    {
        double min_rho_value = std::abs(current_rho - rho_values[0]);
        int min_rho_index = 0;

        for (int index = 1; index < rho_values.size(); index++)
        {
            if (std::abs(current_rho - rho_values[index]) < min_rho_value)
            {
                min_rho_value = std::abs(current_rho - rho_values[index]);
                min_rho_index = index;
            }
        }

        return min_rho_index;
    }
}
