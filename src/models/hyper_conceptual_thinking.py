# /src/models/hyper_conceptual_thinking.py

import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from typing import Dict, List, Tuple
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph
import zenith_hct  # Import the compiled C++ module

class ConceptDiscoveryEngine:
    """
    The Concept Discovery Engine (CDE) is the core of the HCT layer.
    It analyzes the model's internal representations to identify emergent,
    high-value conceptual patterns and stores them persistently in the CKG.
    """
    def __init__(self, ckg: ConceptualKnowledgeGraph, num_clusters: int = 5, discovery_threshold: float = 0.5):
        self.ckg = ckg  # CKG instance
        self.num_clusters = num_clusters
        self.discovery_threshold = discovery_threshold
        self.kmeans_model = None
        self.concept_counter = 0
        # Initialize the C++-side MockCKG instance
        self.cpp_ckg = zenith_hct.MockCKG()


    def analyze_for_new_concepts(self, fused_representation: torch.Tensor, reward: float, domain: str) -> Tuple[float, str | None]:
        """
        Analyzes the fused representation to see if it represents a newly discovered concept.
        
        Args:
            fused_representation: The output of the Conceptual Attention layer.
            reward: The conceptual reward for the current state.
            domain: The current game domain.
        
        Returns:
            A tuple containing a discovery bonus and the name of the discovered concept (if any).
        """
        if reward < self.discovery_threshold:
            return 0.0, None

        # Convert the PyTorch tensor to a NumPy array for C++ interoperability.
        data_np = fused_representation.detach().cpu().numpy().reshape(1, -1)

        # Offload the core HCT calculations to the C++ backend for performance.
        # This function performs complex, parameter-intensive calculations.
        processed_data = zenith_hct.perform_hct_calculations(data_np, self.cpp_ckg)
        
        # Convert the processed data back to a NumPy array for use with MiniBatchKMeans.
        processed_data_np = np.asarray(processed_data)

        if self.kmeans_model is None:
            self.kmeans_model = MiniBatchKMeans(n_clusters=self.num_clusters, n_init=3).fit(processed_data_np)
            self.concept_counter += 1
            concept_name = f"HCT_Concept_{domain}_{self.concept_counter}"

            # Store the discovered concept in the CKG
            self.ckg.add_node(
                concept_name,
                {"type": "discovered_concept", "domain": domain, "reward": reward, "description": f"An emergent pattern in the {domain} domain."}
            )
            return 10.0, concept_name

        new_cluster = self.kmeans_model.predict(processed_data_np)[0]
        distances = self.kmeans_model.transform(processed_data_np)
        min_distance = np.min(distances)

        if min_distance > 1.0:
            self.kmeans_model.partial_fit(processed_data_np)
            self.concept_counter += 1
            concept_name = f"HCT_Concept_{domain}_{self.concept_counter}"

            # Store the new concept in the CKG
            self.ckg.add_node(
                concept_name,
                {"type": "discovered_concept", "domain": domain, "reward": reward, "description": "A significantly different pattern from known concepts."}
            )
            # Add a relationship to the domain
            self.ckg.add_edge(domain, concept_name, "HAS_DISCOVERED_CONCEPT")

            return 5.0, concept_name

        return 0.0, None
