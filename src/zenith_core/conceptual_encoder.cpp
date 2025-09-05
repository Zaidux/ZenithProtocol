// zenith_core/conceptual_encoder.cpp

#include "conceptual_encoder.h"
#include <iostream>

// The conceptual ontology used for tagging words with their conceptual roles.
ConceptualOntology::ConceptualOntology() {
    ontology_map = {
        {"agent", {"person", "robot", "animal", "he", "cat"}},
        {"action", {"eat", "run", "write", "close", "ate"}},
        {"object", {"book", "car", "fish"}},
        {"motion", {"writing", "running"}},
        {"bridge", {"because", "the", "in", "after"}},
        {"property", {"hungry", "tired", "successful", "fragile"}},
        // New: Add socio-linguistic and multimodal properties here as well for C++ side.
        {"socio_linguistic_property", {"formal", "informal", "sarcastic"}},
        {"multimodal_property", {"color:red", "pitch:high"}}
    };

    for (const auto& pair : ontology_map) {
        concept_to_id[pair.first] = next_id++;
        for (const auto& word : pair.second) {
            concept_to_id[word] = next_id++;
        }
    }
}

int ConceptualOntology::get_concept_id(const std::string& concept) {
    if (concept_to_id.find(concept) != concept_to_id.end()) {
        return concept_to_id[concept];
    }
    return -1;
}

std::map<std::string, std::string> ConceptualEncoder::identify_conceptual_roles(const std::string& text, const py::dict& context_py) {
    std::map<std::string, std::string> conceptual_summary;
    // New: Convert Python dict to C++ map.
    std::map<std::string, std::string> context = _py_dict_to_cpp_map(context_py);
    
    // Simple tokenization.
    std::string current_token;
    for (char c : text) {
        if (c == ' ') {
            if (!current_token.empty()) {
                // Check if the token is in the ontology.
                for (const auto& role_pair : ontology.ontology_map) {
                    for (const auto& word : role_pair.second) {
                        if (current_token == word) {
                            conceptual_summary[role_pair.first] = current_token;
                            goto next_token;
                        }
                    }
                }
                // New: Infer socio-linguistic properties.
                if (context.find("tone") != context.end()) {
                    conceptual_summary["tone"] = context["tone"];
                }
                next_token:;
            }
            current_token.clear();
        } else {
            current_token += c;
        }
    }
    return conceptual_summary;
}

py::array_t<double> ConceptualEncoder::encode_conceptual_vector(const py::dict& conceptual_summary_py) {
    size_t embedding_dim = 512;
    auto result_vector = new std::vector<double>(embedding_dim, 0.0);

    // New: Convert conceptual summary from Python dict to C++ map.
    std::map<std::string, std::string> conceptual_summary = _py_dict_to_cpp_map(conceptual_summary_py);

    for (const auto& pair : conceptual_summary) {
        int concept_id = ontology.get_concept_id(pair.first);
        if (concept_id != -1 && concept_id < embedding_dim) {
            (*result_vector)[concept_id] = 1.0;
        }
    }

    auto capsule = py::capsule(result_vector, [](void *p) { delete reinterpret_cast<std::vector<double>*>(p); });
    return py::array_t<double>(result_vector->size(), result_vector->data(), capsule);
}

// New: Helper function to convert a Python dict to a C++ map.
std::map<std::string, std::string> ConceptualEncoder::_py_dict_to_cpp_map(const py::dict& py_dict) {
    std::map<std::string, std::string> cpp_map;
    for (auto item : py_dict) {
        cpp_map[py::str(item.first)] = py::str(item.second);
    }
    return cpp_map;
}

PYBIND11_MODULE(conceptual_encoder_cpp, m) {
    m.doc() = "C++ module for Zenith Conceptual Encoder.";
    py::class_<ConceptualOntology>(m, "ConceptualOntology").def(py::init<>());
    py::class_<ConceptualEncoder>(m, "ConceptualEncoder")
        .def(py::init<>())
        .def("identify_conceptual_roles", &ConceptualEncoder::identify_conceptual_roles)
        .def("encode_conceptual_vector", &ConceptualEncoder::encode_conceptual_vector);
    m.def("get_encoder", []() { return ConceptualEncoder(); }, "Returns an instance of the ConceptualEncoder.");
}
