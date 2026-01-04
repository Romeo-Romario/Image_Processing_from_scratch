#pragma once
#include <thread>
#include <iostream>
#include <vector>
#include <utility>
#include <memory>
#include <ranges>
#include "../../additional_modules/include/matrix_convert.hpp"
#include "additional_functions.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using std::cout;
using std::endl;
using std::vector;
using Matrix = vector<vector<double>>;

class CannyEdgeDetector
{
private:
    int rows, cols;
    Matrix grey_image,
        kernel_matrix,
        convolved_image,
        dI_dX, dI_dY,
        gradient_magnitued,
        grad_oreo,
        rounded_grad_oreo,
        grad_mag2,
        thresholded_img,
        hyst_img;

    unsigned int n_threads = std::max(1u, std::thread::hardware_concurrency());
    void convolve_image(double sigma, bool output = false);
    void accumulate_image_gradients();
    void accumulate_image_gradient_orientation();
    void accumulate_non_max_suppresion();
    void accumulate_thresholded_img(double maxx, double minn, double meann, double hight_treshold, double low_treshold);
    py::array_t<double> accumulate_hysteresis_img();

public:
    CannyEdgeDetector() = default;
    CannyEdgeDetector(const CannyEdgeDetector &el);
    CannyEdgeDetector(const Matrix &input_image);
    CannyEdgeDetector(const py::array_t<double> &input_image);

    py::array_t<double> get_canny_img(const py::array_t<double> &input_image,
                                      double sigma = 1.0,
                                      double hight_threshold = 0.4,
                                      double low_threshold = 0.06);

    const Matrix &get_convolved_image() const;
    vector<py::array_t<double>> get_image_gradients();
    vector<py::array_t<double>> get_image_gradient_orientation();
    py::array_t<double> get_non_max_suppresion();
    py::array_t<double> get_thresholded_img();
    py::array_t<double> get_hysteresis_img();
};