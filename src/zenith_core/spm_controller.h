// zenith_core/spm_controller.h

#ifndef SPM_CONTROLLER_H
#define SPM_CONTROLLER_H

#include <vector>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class SPMController {
public:
    SPMController();
    
    // Dynamically allocates parameters to handle multiple tasks in parallel.
    // This is a mock function that simulates the process.
    void allocate_for_tasks(const std::vector<std::string>& domains);
    
    // Runs a simulation for a specific task.
    // The core of the SPM feature.
    py::array_t<double> run_parallel_simulation(py::array_t<double> input_data, const std::string& domain);
};

#endif // SPM_CONTROLLER_H
