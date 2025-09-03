// zenith_core/zenith_neural_calculations.h

#ifndef ZENITH_NEURAL_CALCULATIONS_H
#define ZENITH_NEURAL_CALCULATIONS_H

#include <vector>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Forward declaration of the MockCKG class to prevent compilation errors
class MockCKG;

// Function declaration for performing Hyper-Conceptual Thinking (HCT) calculations.
// This function takes a numpy array and a reference to a MockCKG object.
py::array_t<double> perform_hct_calculations(py::array_t<double> conceptual_vector, MockCKG& ckg);

// A simplified representation of the Conceptual Knowledge Graph (CKG) for C++.
// The full implementation is in the .cpp file.
class MockCKG {
public:
    // A public function declaration to get conceptual properties.
    std::vector<double> get_conceptual_properties(const std::string& concept_name);
};

#endif // ZENITH_NEURAL_CALCULATIONS_H

