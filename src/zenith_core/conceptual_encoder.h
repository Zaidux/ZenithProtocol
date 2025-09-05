// zenith_core/conceptual_encoder.h

#ifndef CONCEPTUAL_ENCODER_H
#define CONCEPTUAL_ENCODER_H

#include <string>
#include <vector>
#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// A class to represent the simplified Conceptual Ontology.
class ConceptualOntology {
public:
    std::map<std::string, std::vector<std::string>> ontology_map;
    std::map<std::string, int> concept_to_id;
    int next_id = 0;

    ConceptualOntology();

    int get_concept_id(const std::string& concept);
};

// The core Conceptual Encoder class.
class ConceptualEncoder {
public:
    ConceptualOntology ontology;

    // New: The function now accepts a context map.
    std::map<std::string, std::string> identify_conceptual_roles(const std::string& text, const py::dict& context_py);

    py::array_t<double> encode_conceptual_vector(const py::dict& conceptual_summary_py);

private:
    // New: Helper function to convert a Python dict to a C++ map.
    std::map<std::string, std::string> _py_dict_to_cpp_map(const py::dict& py_dict);
};

// Function declaration for the Python module.
void init_conceptual_encoder(py::module &m);

#endif // CONCEPTUAL_ENCODER_H
