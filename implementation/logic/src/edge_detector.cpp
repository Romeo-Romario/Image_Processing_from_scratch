#include "../include/edge_detector.hpp"

EdgeDetector::EdgeDetector(const EdgeDetector &el)
{
    grey_image = el.grey_image;
    kernel_matrix = el.kernel_matrix;
    convolved_image = el.convolved_image;
    dI_dX = el.dI_dX;
    dI_dY = el.dI_dY;
    dI2 = el.dI2;
}
EdgeDetector::EdgeDetector(const vector<vector<double>> &input_image)
{
    grey_image = input_image;
    rows = grey_image.size();
    cols = grey_image[0].size();
    chunk_size = rows / n_threads;
}
EdgeDetector::EdgeDetector(const py::array_t<double> &input_image)
{
    auto buf = input_image.request();
    if (buf.ndim != 2)
        throw std::runtime_error("Input image must be 2D");
    rows = buf.shape[0];
    cols = buf.shape[1];
    auto ptr = static_cast<double *>(buf.ptr);
    grey_image = vector(rows, std::vector<double>(cols));
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            grey_image[i][j] = ptr[i * cols + j];
    chunk_size = rows / n_threads;
}

vector<vector<double>> EdgeDetector::convolve_image(double sigma, bool output)
{
    // Set new Kernel matrix
    kernel_matrix = gaussian_kernel(sigma);
    if (output)
    {
        cout << "Kernel matrix\n";
        for (int i = 0; i < kernel_matrix.size(); i++)
        {
            for (int j = 0; j < kernel_matrix[i].size(); j++)
            {
                cout << kernel_matrix[i][j] << " ";
            }
            cout << endl;
        }
    }
    convolved_image = vector(rows, std::vector<double>(cols, 0.0));
    vector<std::thread> threads;
    for (int t = 0; t < n_threads; ++t)
    {
        int start = t * chunk_size;
        int end = (t == n_threads - 1) ? rows : start + chunk_size;
        threads.emplace_back(convolve_chunk, std::ref(grey_image), std::ref(kernel_matrix), std::ref(convolved_image), start, end);
    }
    for (auto &t : threads)
        t.join();

    return convolved_image;
}

vector<py::array_t<double>> EdgeDetector::generate_matrixes()
{
    using Matrix = vector<vector<double>>;
    // These contain the RESULTS (Full Images)
    std::vector<Matrix *> matrix_ptrs = {&dI_dX, &dI_dY, &dI2};
    // These contain the INPUT KERNELS (3x3)
    vector<Matrix> derivative_matrixes = {dI_dx, dI_dy, d2I_dxdy};

    // FIX 1: Move thread vector INSIDE the loop (or clear it every time)
    // so we don't try to join the same thread twice.
    for (int matrix_index = 0; matrix_index < matrix_ptrs.size(); matrix_index++)
    {
        // Resize the destination matrix to match the image size before processing!
        // (Ensure dI_dX, etc. have the right size or convolve_chunk will crash)
        if (matrix_ptrs[matrix_index]->empty())
        {
            matrix_ptrs[matrix_index]->resize(rows, std::vector<double>(cols, 0.0));
        }

        vector<std::thread> threads;

        for (int t = 0; t < n_threads; ++t)
        {
            int start = t * chunk_size;
            int end = (t == n_threads - 1) ? rows : start + chunk_size;

            threads.emplace_back(convolve_chunk,
                                 std::ref(convolved_image),
                                 std::ref(derivative_matrixes[matrix_index]),
                                 std::ref(*matrix_ptrs[matrix_index]),
                                 start, end);
        }
        for (auto &t : threads)
            t.join();
    }

    // Part 2

    vector<std::thread> threads;

    for (int t = 0; t < n_threads; ++t)
    {
        int start = t * chunk_size;
        int end = (t == n_threads - 1) ? rows : start + chunk_size;

        threads.emplace_back(chunk_zero_crossing,
                             std::ref(*matrix_ptrs[2]),
                             start, end);
    }
    for (auto &t : threads)
        t.join();

    // Part 3
    std::vector<py::array_t<double>> result;
    result.reserve(3);

    for (int matrix_index = 0; matrix_index < 3; matrix_index++)
    {
        py::array_t<double> py_array({rows, cols});

        auto accessor = py_array.mutable_unchecked<2>();

        // FIX 3: Read from the CALCULATED image (*matrix_ptrs), not the 3x3 kernel
        Matrix &source_data = *matrix_ptrs[matrix_index];

        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                // Copy data from C++ vector to Numpy
                accessor(i, j) = source_data[i][j];
            }
        }

        // Add the filled array to the list
        result.push_back(py_array);

        // Optional: Clear C++ memory if you don't need it anymore
        matrix_ptrs[matrix_index]->clear();
    }

    return result;
}