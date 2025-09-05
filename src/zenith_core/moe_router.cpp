// zenith_core/moe_router.cpp

#include "moe_router.h"
#include <cmath>
#include <iostream>
#include <numeric>

std::vector<double> softmax(const std::vector<double>& logits) {
    std::vector<double> exp_logits;
    double sum_exp = 0.0;
    for (double logit : logits) {
        double exp_val = std::exp(logit);
        exp_logits.push_back(exp_val);
        sum_exp += exp_val;
    }
    std::vector<double> probabilities;
    for (double exp_val : exp_logits) {
        probabilities.push_back(exp_val / sum_exp);
    }
    return probabilities;
}

py::dict convert_map_to_py_dict(const std::map<std::string, std::vector<std::string>>& map) {
    py::dict result;
    for (const auto& pair : map) {
        py::list val_list;
        for (const auto& item : pair.second) {
            val_list.append(item);
        }
        result[py::str(pair.first)] = val_list;
    }
    return result;
}

py::array_t<double> ConceptualContext::get_context_vector(int num_experts) {
    auto result_vector = new std::vector<double>(num_experts, 0.0);
    std::map<std::string, int> concept_expert_map = {
        {"chess", 0}, {"game", 1}, {"tetris", 2}, {"finance", 3}, {"law", 4}, {"medicine", 5}, {"strategy", 0}, {"knowledge", 1}
    };

    for (const auto& pair : context_map) {
        for (const auto& concept : pair.second) {
            if (concept_expert_map.find(concept) != concept_expert_map.end()) {
                (*result_vector)[concept_expert_map[concept]] += 1.0;
            }
        }
    }

    auto capsule = py::capsule(result_vector, [](void *p) { delete reinterpret_cast<std::vector<double>*>(p); });
    return py::array_t<double>(result_vector->size(), result_vector->data(), capsule);
}

ConceptualAwareRouter::ConceptualAwareRouter(int input_dim, int num_experts, int top_k)
    : input_dim_(input_dim), num_experts_(num_experts), top_k_(top_k) {
    router_weights.resize(input_dim * num_experts);
    for (size_t i = 0; i < router_weights.size(); ++i) {
        router_weights[i] = static_cast<double>(rand()) / RAND_MAX * 2.0 - 1.0;
    }
}

py::array_t<double> ConceptualAwareRouter::route(py::array_t<double> input_tensor, const ConceptualContext& context) {
    py::buffer_info buf = input_tensor.request();
    auto input_ptr = static_cast<double*>(buf.ptr);

    py::array_t<double> context_vector = context.get_context_vector(num_experts_);
    auto context_ptr = static_cast<double*>(context_vector.request().ptr);

    std::vector<double> gate_logits(num_experts_, 0.0);
    for (int i = 0; i < num_experts_; ++i) {
        double dot_product = 0.0;
        for (int j = 0; j < input_dim_; ++j) {
            dot_product += input_ptr[j] * router_weights[i * input_dim_ + j];
        }
        gate_logits[i] = dot_product + context_ptr[i];
    }

    std::vector<double> weights = softmax(gate_logits);

    std::vector<std::pair<double, int>> ranked_experts;
    for (int i = 0; i < num_experts_; ++i) {
        ranked_experts.push_back({weights[i], i});
    }
    std::sort(ranked_experts.rbegin(), ranked_experts.rend());

    auto top_k_indices = new std::vector<double>(top_k_);
    for (int i = 0; i < top_k_; ++i) {
        (*top_k_indices)[i] = ranked_experts[i].second;
    }

    auto capsule = py::capsule(top_k_indices, [](void *p) { delete reinterpret_cast<std::vector<double>*>(p); });
    return py::array_t<double>(top_k_indices->size(), top_k_indices->data(), capsule);
}

double ConceptualAwareRouter::calculate_load_balancing_loss(const py::array_t<double>& weights, const py::array_t<double>& top_k_weights, const py::array_t<double>& top_k_indices) {
    py::buffer_info weights_buf = weights.request();
    auto weights_ptr = static_cast<double*>(weights_buf.ptr);

    py::buffer_info indices_buf = top_k_indices.request();
    auto indices_ptr = static_cast<double*>(indices_buf.ptr);

    std::vector<double> avg_router_prob(num_experts_, 0.0);
    std::vector<double> fraction_of_tokens_routed(num_experts_, 0.0);
    double total_tokens = 1.0;

    for (int i = 0; i < num_experts_; ++i) {
        avg_router_prob[i] = weights_ptr[i];
    }

    for (int i = 0; i < top_k_; ++i) {
        int expert_idx = static_cast<int>(indices_ptr[i]);
        fraction_of_tokens_routed[expert_idx] += 1.0 / total_tokens;
    }

    double load_loss = 0.0;
    for (int i = 0; i < num_experts_; ++i) {
        load_loss += avg_router_prob[i] * fraction_of_tokens_routed[i];
    }

    return load_loss;
}

PYBIND11_MODULE(moe_router_cpp, m) {
    m.doc() = "C++ module for Zenith Mixture of Experts Router.";

    py::class_<ConceptualContext>(m, "ConceptualContext")
        .def(py::init<>())
        .def_readwrite("context_map", &ConceptualContext::context_map);

    py::class_<ConceptualAwareRouter>(m, "ConceptualAwareRouter")
        .def(py::init<int, int, int>())
        .def("route", &ConceptualAwareRouter::route)
        .def("calculate_load_balancing_loss", &ConceptualAwareRouter::calculate_load_balancing_loss);
}

