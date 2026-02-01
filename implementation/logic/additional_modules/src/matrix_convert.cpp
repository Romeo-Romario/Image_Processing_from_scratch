#include "../include/matrix_convert.hpp"

namespace additional_modules
{
    namespace matrix_converter
    {
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

        Matrix convert_numpy_array_to_matrix(const py::array_t<double> &numpy_matrix)
        {
            Matrix grey_image;

            auto buf = numpy_matrix.request();
            int rows = buf.shape[0];
            int cols = buf.shape[1];
            if (buf.ndim != 2)
                throw std::runtime_error("Input image must be 2D");
            auto ptr = static_cast<double *>(buf.ptr);
            grey_image = vector(rows, std::vector<double>(cols));
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    grey_image[i][j] = ptr[i * cols + j];

            return grey_image;
        }

        py::array_t<double> convert_matrix_to_numpy_array(const Matrix &matrix)
        {
            int rows = matrix.size();
            int cols = matrix[0].size();

            py::array_t<double> py_array({rows, cols});

            auto accessor = py_array.mutable_unchecked<2>();

            const Matrix &source_data = matrix;

            for (size_t i = 0; i < rows; ++i)
            {
                for (size_t j = 0; j < cols; ++j)
                {
                    accessor(i, j) = source_data[i][j];
                }
            }

            return py_array;
        }

        Matrix3D convert_numpy_array_to_3d_matrix(const py::array_t<double> &numpy_array)
        {
            auto buf = numpy_array.request();

            std::cout << "1\n";
            size_t rows = buf.shape[0];
            size_t cols = buf.shape[1];
            size_t channels = buf.shape[2];

            auto ptr = static_cast<double *>(buf.ptr);
            std::cout << "2\n";
            Matrix3D image_3d(rows, vector<vector<double>>(cols, vector<double>(channels)));
            std::cout << "3\n";
            for (size_t i = 0; i < rows; ++i)
            {
                for (size_t j = 0; j < cols; ++j)
                {
                    for (size_t k = 0; k < channels; ++k)
                    {
                        size_t index = (i * cols * channels) + (j * channels) + k;
                        image_3d[i][j][k] = ptr[index];
                    }
                }
            }

            std::cout << "4\n";
            return image_3d;
        }
    }
}
