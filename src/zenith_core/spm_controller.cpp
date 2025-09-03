// zenith_core/spm_controller.cpp

#include "spm_controller.h"
#include <iostream>
#include <random>

SPMController::SPMController() {
    std::cout << "SPM Controller initialized in C++." << std::endl;
}

void SPMController::allocate_for_tasks(const std::vector<std::string>& domains) {
    std::cout << "Dynamically allocating computational parameters for tasks in domains: ";
    for (const auto& domain : domains) {
        std::cout << domain << " ";
    }
    std::cout << std::endl;
}

py::array_t<double> SPMController::run_parallel_simulation(py::array_t<double> input_data, const std::string& domain) {
    std::cout << "Running parallel simulation for domain: " << domain << std::endl;
    
    py::buffer_info buf = input_data.request();
    double* ptr = static_cast<double*>(buf.ptr);
    size_t vector_size = buf.size;
    
    auto result_vector = new std::vector<double>(vector_size, 0.0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    // Simulate domain-specific processing.
    if (domain == "tetris") {
        for (size_t i = 0; i < vector_size; ++i) {
            (*result_vector)[i] = ptr[i] * dis(gen) * 2.5;
        }
    } else if (domain == "chess") {
        for (size_t i = 0; i < vector_size; ++i) {
            (*result_vector)[i] = ptr[i] * dis(gen) * 1.8;
        }
    } else {
        for (size_t i = 0; i < vector_size; ++i) {
            (*result_vector)[i] = ptr[i] * dis(gen);
        }
    }
    
    auto capsule = py::capsule(result_vector, [](void *p) { delete reinterpret_cast<std::vector<double>*>(p); });
    return py::array_t<double>(result_vector->size(), result_vector->data(), capsule);
}

// Expose the class to Python
PYBIND11_MODULE(spm_controller_cpp, m) {
    py::class_<SPMController>(m, "SPMController")
        .def(py::init<>())
        .def("allocate_for_tasks", &SPMController::allocate_for_tasks)
        .def("run_parallel_simulation", &SPMController::run_parallel_simulation);
}
