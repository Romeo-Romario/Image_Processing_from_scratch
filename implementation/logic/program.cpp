#include <iostream>
#include <vector>
#include "include/additional_functions.hpp"
using namespace std;

template <typename T>
void print_2_vector(const vector<vector<T>> &element)
{
    for (int i = 0; i < element.size(); i++)
    {
        for (int j = 0; j < element[i].size(); j++)
        {
            cout << element[i][j] << " ";
        }
        cout << endl;
    }
}

int main()
{
    vector<vector<double>> kernel_matrix = gaussian_kernel(0.2);
    print_2_vector<double>(kernel_matrix);
}