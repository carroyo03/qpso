#include <omp.h>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
constexpr double PI = 3.14159265358979323846;

// Helper function to get the flattened size of the array (excluding the first dimension for 2D+ arrays)
size_t get_position_size(const py::array_t<double>& position) {
    auto shape = position.shape();
    size_t ndim = position.ndim();
    if (ndim == 0) {
        throw std::invalid_argument("Input array must have at least one dimension");
    }
    size_t size = 1;
    for (size_t i = (ndim == 1 ? 0 : 1); i < ndim; ++i) {
        size *= shape[i];
    }
    return size;
}

double rastrigin_parallel(py::array_t<double> position) {
    auto buf = position.request();
    double* data = static_cast<double*>(buf.ptr);
    size_t n = get_position_size(position);

    double result = 0.0;

    #pragma omp parallel for reduction(+:result)
    for (size_t i = 0; i < n; ++i) {
        double xi = data[i];
        result += xi * xi - 10.0 * std::cos(2.0 * PI * xi);
    }

    return 10.0 * n + result;
}

double ackley_parallel(py::array_t<double> position) {
    auto buf = position.request();
    double* data = static_cast<double*>(buf.ptr);
    size_t n = get_position_size(position);

    double sum1 = 0.0;
    double sum2 = 0.0;
    double a = 20.0;
    double b = 0.2;
    double c = 2.0 * PI;

    #pragma omp parallel for reduction(+:sum1, sum2)
    for (size_t i = 0; i < n; ++i) {
        double xi = data[i];
        sum1 += xi * xi;
        sum2 += std::cos(c * xi);
    }

    double result = -a * std::exp(-b * std::sqrt(sum1 / n)) - std::exp(sum2 / n) + a + std::exp(1.0);
    return result;
}

double rosenbrock_parallel(py::array_t<double> position) {
    auto buf = position.request();
    double* data = static_cast<double*>(buf.ptr);
    size_t n = get_position_size(position);

    double result = 0.0;

    #pragma omp parallel for reduction(+:result)
    for (size_t i = 0; i < n - 1; ++i) {
        double xi = data[i];
        double xi1 = data[i + 1];
        result += 100.0 * std::pow(xi1 - xi * xi, 2) + std::pow(1.0 - xi, 2);
    }

    return result;
}

py::array_t<double> evaluate_swarm(py::array_t<double> positions, std::string function) {
    // Get buffer info
    auto buf = positions.request();
    double* data = static_cast<double*>(buf.ptr);
    size_t ndim = buf.ndim; 
    std::vector<ssize_t> shape = buf.shape;

    // Determine number of particles and dimension
    size_t num_particles = (ndim <= 1) ? 1 : shape[0];
    size_t dim = get_position_size(positions);


    // Create result array
    auto result = py::array_t<double>(num_particles);
    auto result_unchecked = result.template mutable_unchecked<1>();

    #pragma omp parallel for
    for (size_t i = 0; i < num_particles; ++i) {
        // Create a view of the particle's position
        py::array_t<double> position;
        if (ndim <= 1) {
            // For 1D input, use the entire array
            position = positions;
        } else {
            // For 2D+ input, create a view of the i-th particle's position
            std::vector<ssize_t> pos_shape(shape.begin() + 1, shape.end());
            position = py::array_t<double>(pos_shape, {sizeof(double)}, data + i * dim, positions);
        }

        if (function == "rastrigin") {
            result_unchecked(i) = rastrigin_parallel(position);
        } else if (function == "ackley") {
            result_unchecked(i) = ackley_parallel(position);
        } else if (function == "rosenbrock") {
            result_unchecked(i) = rosenbrock_parallel(position);
        } else {
            throw std::invalid_argument("Unknown function: " + function);
        }
    }

    return result;
}

PYBIND11_MODULE(pso_core, m) {
    m.doc() = "C++ implementation with OpenMP for PSO functions";
    m.def("rastrigin_parallel", &rastrigin_parallel, "Evaluate Rastrigin function with OpenMP");
    m.def("ackley_parallel", &ackley_parallel, "Evaluate Ackley function with OpenMP");
    m.def("rosenbrock_parallel", &rosenbrock_parallel, "Evaluate Rosenbrock function with OpenMP");
    m.def("evaluate_swarm", &evaluate_swarm, "Evaluate particle swarm positions with OpenMP");
}