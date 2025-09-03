// zenith_core/eom_calculator.cpp

#include "eom_calculator.h"
#include <cmath>
#include <iostream>

EoMCalculator::EoMCalculator() {
    std::cout << "EoM Calculator initialized in C++." << std::endl;
}

double EoMCalculator::calculate_eom_bonus(py::array_t<double> last_fused_rep, py::array_t<double> current_fused_rep, double eom_weight) {
    py::buffer_info last_buf = last_fused_rep.request();
    py::buffer_info current_buf = current_fused_rep.request();

    if (last_buf.size != current_buf.size) {
        throw std::runtime_error("Input vectors must have the same size.");
    }

    double* last_ptr = static_cast<double*>(last_buf.ptr);
    double* current_ptr = static_cast<double*>(current_buf.ptr);
    size_t size = last_buf.size;

    double conceptual_change_squared = 0.0;
    for (size_t i = 0; i < size; ++i) {
        double diff = current_ptr[i] - last_ptr[i];
        conceptual_change_squared += diff * diff;
    }
    
    double conceptual_change = std::sqrt(conceptual_change_squared);
    double eom_bonus = eom_weight * conceptual_change;

    return eom_bonus;
}

// Expose the class to Python
PYBIND11_MODULE(eom_calculator_cpp, m) {
    py::class_<EoMCalculator>(m, "EoMCalculator")
        .def(py::init<>())
        .def("calculate_eom_bonus", &EoMCalculator::calculate_eom_bonus);
}
