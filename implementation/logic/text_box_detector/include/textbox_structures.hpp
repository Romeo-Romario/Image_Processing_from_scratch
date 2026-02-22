#pragma once
#include <vector>

using std::vector;
using Matrix = vector<vector<double>>;

struct Point
{
    int x;
    int y;
};

struct TextRow
{
    Matrix text_matrix;
    int y_start;
    int y_end;
    int x_start;
    int x_end;
    int index_in_original_img;

    // Additional fields for precise analysis
    vector<double> _1d_function;
    vector<int> zero_sep_points;
    vector<int> potetional_zero_sep_points;
    vector<std::pair<Point, Point>> symbols_limits;
};

struct SymbolBox
{
    Matrix symbol_matrix;
    int y_start;
    int y_end;
    int x_start;
    int x_end;
};
