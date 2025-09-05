# /src/models/hyper_conceptual_thinking.py

import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from typing import Dict, List, Tuple
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph
import zenith_hct
from .architectural_reconfiguration_layer import ArchitecturalReconfigurationLayer # New Import

class ConceptDiscoveryEngine:
    def __init__(self, ckg: ConceptualKnowledgeGraph, model: torch.nn.Module, num_clusters: int = 5, discovery_threshold: float = 0.5): # New 'model' dependency
        self.ckg = ckg
        self.model = model
        self.num_clusters = num_clusters
        self.discovery_threshold = discovery_threshold
        self.kmeans_model = None
        self.concept_counter = 0
        self.cpp_ckg = zenith_hct.MockCKG()
        self.relationship_types_counter = 0
        self.arch_layer = ArchitecturalReconfigurationLayer(model, ckg) # New instance

    def analyze_for_new_concepts(self, fused_representation: torch.Tensor, confidence_score: float, reward: float, domain: str) -> Tuple[float, str | None]:
        if confidence_score < 0.3 or reward > self.discovery_threshold:
            print(f"HCT triggered. Confidence: {confidence_score}, Reward: {reward}")
            data_np = fused_representation.detach().cpu().numpy().reshape(1, -1)
            processed_data_np = np.asarray(zenith_hct.perform_hct_calculations(data_np, self.cpp_ckg))

            if self.kmeans_model is None:
                self.kmeans_model = MiniBatchKMeans(n_clusters=self.num_clusters, n_init=3).fit(processed_data_np)
                self.concept_counter += 1
                concept_name = f"HCT_Concept_{domain}_{self.concept_counter}"
                self.ckg.add_node(
                    concept_name,
                    {"type": "discovered_concept", "domain": domain, "reward": reward, "description": f"An emergent pattern in the {domain} domain."}
                )
                
                # New: Propose architectural upgrade if a major, novel concept is discovered.
                if self.concept_counter == 1: # A new, major conceptual discovery
                    self._propose_architectural_upgrade("add_new_expert")
                
                return 10.0, concept_name

            new_cluster = self.kmeans_model.predict(processed_data_np)[0]
            distances = self.kmeans_model.transform(processed_data_np)
            min_distance = np.min(distances)

            if min_distance > 1.0:
                self.kmeans_model.partial_fit(processed_data_np)
                self.concept_counter += 1
                concept_name = f"HCT_Concept_{domain}_{self.concept_counter}"
                new_relationship_name = f"HCT_Relationship_{self.relationship_types_counter}"
                self.ckg.propose_new_relationship_type(new_relationship_name, "An emergent, novel relationship discovered by HCT.")
                self.relationship_types_counter += 1
                self.ckg.add_node(
                    concept_name,
                    {"type": "discovered_concept", "domain": domain, "reward": reward, "description": "A significantly different pattern from known concepts."}
                )
                self.ckg.add_edge(domain, concept_name, new_relationship_name)
                return 5.0, concept_name
        return 0.0, None
    
    # New Method: Proposes an architectural upgrade and logs it to the CKG.
    def _propose_architectural_upgrade(self, upgrade_type: str):
        proposal_id = f"proposal_{self.ckg.proposal_counter}"
        proposal = {
            "proposal_id": proposal_id,
            "proposer": "HCT",
            "type": upgrade_type,
            "predicted_impact": self.arch_layer.predict_impact(upgrade_type),
            "proposed_changes": self.arch_layer.get_proposed_changes(upgrade_type),
            "status": "pending_human_review",
            "timestamp": datetime.now().isoformat()
        }
        self.ckg.add_node(proposal_id, proposal)
        self.ckg.add_edge("HCT", proposal_id, "PROPOSED_UPGRADE")
        print(f"\n[HCT] Proposed a new architectural upgrade: '{upgrade_type}'")
        print(f"Waiting for human review via the Explainability Module.")

