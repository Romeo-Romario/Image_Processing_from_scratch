#pragma once
#include <vector>

using std::vector;
using Matrix = vector<vector<double>>;

struct TextRow
{
    Matrix text_matrix;
    vector<double> _1d_function;
    int y_start;
    int y_end;
    int x_start;
    int x_end;
    int index_in_original_img;
};
