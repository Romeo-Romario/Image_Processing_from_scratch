#include "../include/matrix_converter.hpp"

std::vector<py::array_t<double>> convert_matrixes_pointers_to_numpy_array(const vector<Matrix *> &matrixes)
{
    int number_of_matrixes = matrixes.size();
    std::vector<py::array_t<double>> result;
    result.reserve(number_of_matrixes);

    int rows = matrixes[0]->size();
    int cols = (*matrixes[0])[0].size();

    for (int matrix_index = 0; matrix_index < number_of_matrixes; matrix_index++)
    {
        py::array_t<double> py_array({rows, cols});

        auto accessor = py_array.mutable_unchecked<2>();

        Matrix &source_data = *matrixes[matrix_index];

        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                accessor(i, j) = source_data[i][j];
            }
        }

        result.push_back(py_array);
    }

    return result;
}

std::vector<py::array_t<double>> convert_matrixes_to_numpy_array(const vector<Matrix> &matrixes)
{
    std::vector<py::array_t<double>> result;
    int number_of_matrixes = matrixes.size();
    result.reserve(number_of_matrixes);

    int rows = matrixes[0].size();
    int cols = matrixes[0][0].size();

    for (int matrix_index = 0; matrix_index < number_of_matrixes; matrix_index++)
    {
        py::array_t<double> py_array({rows, cols});

        auto accessor = py_array.mutable_unchecked<2>();

        const Matrix &source_data = matrixes[matrix_index];

        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                accessor(i, j) = source_data[i][j];
            }
        }

        result.push_back(py_array);
    }

    return result;
}