// zenith_core/eom_calculator.h

#ifndef EOM_CALCULATOR_H
#define EOM_CALCULATOR_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class EoMCalculator {
public:
    EoMCalculator();
    
    // Calculates the Energy of Movement bonus by comparing two conceptual representations.
    double calculate_eom_bonus(py::array_t<double> last_fused_rep, py::array_t<double> current_fused_rep, double eom_weight);
};

#endif // EOM_CALCULATOR_H
