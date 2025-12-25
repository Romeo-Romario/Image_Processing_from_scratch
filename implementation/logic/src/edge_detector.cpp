#include "../include/edge_detector.hpp"
#include "../include/matrix_converter.hpp"

EdgeDetector::EdgeDetector(const EdgeDetector &el)
{
    grey_image = el.grey_image;
    kernel_matrix = el.kernel_matrix;
    convolved_image = el.convolved_image;
    dI_dX = el.dI_dX;
    dI_dY = el.dI_dY;
    gradient_magnitued = el.gradient_magnitued;
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

vector<py::array_t<double>> EdgeDetector::get_image_gradients()
{
    /*
        Based on Sobel 3x3 matrixes calculates dI/dx, dI/dy and magnitued matrixes
    */
    std::vector<Matrix *> matrix_ptrs = {&dI_dX, &dI_dY, &gradient_magnitued};
    vector<Matrix> derivative_matrixes = {dI_dx, dI_dy};

    for (int matrix_index = 0; matrix_index < matrix_ptrs.size() - 1; matrix_index++)
    {
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

    if (matrix_ptrs[2]->empty())
    {
        matrix_ptrs[2]->resize(rows, std::vector<double>(cols, 0.0));
    }

    vector<std::thread> threads;

    for (int t = 0; t < n_threads; ++t)
    {
        try
        {
            int start = t * chunk_size;
            int end = (t == n_threads - 1) ? rows : start + chunk_size;

            threads.emplace_back(chunk_gradient_magnitute,
                                 std::ref(*matrix_ptrs[0]),
                                 std::ref(*matrix_ptrs[1]),
                                 std::ref(*matrix_ptrs[2]),
                                 start, end);
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
        }
    }
    for (auto &t : threads)
        t.join();

    std::vector<py::array_t<double>> result = convert_matrixes_pointers_to_numpy_array(std::move(matrix_ptrs));
    return result;
}

vector<py::array_t<double>> EdgeDetector::get_image_gradient_orientation()
{
    Matrix small_val(rows, vector<double>(cols, 0.0001));

    return std::vector<py::array_t<double>>();
}