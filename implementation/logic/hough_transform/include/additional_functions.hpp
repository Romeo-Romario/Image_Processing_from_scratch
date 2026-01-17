#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using std::cout;
using std::endl;
using std::vector;

namespace additional_functions
{
    int rho_array_argmin(vector<double> rho_values, double current_rho = 0);

    template <typename T>
    std::vector<T> arange(T start, T stop, T step)
    {
        if (step == 0)
            return {};
        size_t num_elements = static_cast<size_t>(std::ceil((stop - start) / step));
        std::vector<T> vec(num_elements);
        T current_val = start;
        for (size_t i = 0; i < num_elements; ++i)
        {
            vec[i] = current_val;
            current_val += step;
        }
        return vec;
    }

    template <typename T>
    void print_vec(vector<T> v, std::string vec_name)
    {
        cout << "Vector " << vec_name << " size: " << v.size() << endl;
        cout << "Elements: { ";
        for (auto el : v)
        {
            cout << el << " ";
        }
        cout << "}" << endl;
    }
}