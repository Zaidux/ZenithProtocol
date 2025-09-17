# /src/models/hyper_conceptual_thinking.py

"""
Enhanced Hyper-Conceptual Thinking with Cross-Domain Discovery
=============================================================
Now supports cross-domain concept discovery and architectural innovation.
"""

import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph
import zenith_hct
from .architectural_reconfiguration_layer import ArchitecturalReconfigurationLayer

class ConceptDiscoveryEngine:
    def __init__(self, ckg: ConceptualKnowledgeGraph, model: torch.nn.Module, 
                 num_clusters: int = 8, discovery_threshold: float = 0.6):
        self.ckg = ckg
        self.model = model
        self.num_clusters = num_clusters
        self.discovery_threshold = discovery_threshold
        self.kmeans_model = None
        self.dbscan_model = DBSCAN(eps=0.5, min_samples=2)  # Added density-based clustering
        self.concept_counter = 0
        self.cpp_ckg = zenith_hct.MockCKG()
        self.relationship_types_counter = 0
        self.arch_layer = ArchitecturalReconfigurationLayer(model, ckg)
        
        # Cross-domain discovery tracking
        self.cross_domain_insights = {}
        self.innovation_score = 0.0

    def analyze_for_new_concepts(self, fused_representation: torch.Tensor, 
                               confidence_score: float, reward: float, 
                               domain: str) -> Tuple[float, Optional[str]]:
        """
        Enhanced concept discovery with cross-domain capabilities.
        """
        # Check discovery conditions with adaptive thresholds
        should_discover = (
            confidence_score < 0.4 or 
            reward > self.discovery_threshold or
            self._is_novel_pattern(fused_representation, domain)
        )
        
        if not should_discover:
            return 0.0, None

        print(f"HCT triggered. Confidence: {confidence_score}, Reward: {reward}")
        data_np = fused_representation.detach().cpu().numpy().reshape(1, -1)
        processed_data_np = np.asarray(zenith_hct.perform_hct_calculations(data_np, self.cpp_ckg))

        # Try multiple clustering approaches
        concept_name, bonus = self._cluster_and_discover(processed_data_np, domain, reward)
        
        # Cross-domain pattern analysis
        cross_domain_bonus = self._analyze_cross_domain_patterns(processed_data_np, domain)
        total_bonus = bonus + cross_domain_bonus
        
        return total_bonus, concept_name

    def _cluster_and_discover(self, data: np.ndarray, domain: str, reward: float) -> Tuple[Optional[str], float]:
        """Enhanced clustering with multiple algorithms."""
        # First, try density-based clustering for outlier detection
        dbscan_labels = self.dbscan_model.fit_predict(data)
        unique_labels = np.unique(dbscan_labels)
        
        # If outliers found (-1 label), they might be novel concepts
        if -1 in unique_labels and len(unique_labels) > 1:
            outlier_mask = dbscan_labels == -1
            if np.any(outlier_mask):
                return self._create_novel_concept(data[outlier_mask], domain, reward, "density_based")

        # Fall back to K-means for general clustering
        if self.kmeans_model is None:
            self.kmeans_model = MiniBatchKMeans(n_clusters=self.num_clusters, n_init=3)
            self.kmeans_model.fit(data)
            return self._create_novel_concept(data, domain, reward, "initial_cluster")

        # Existing K-means logic with enhancements
        distances = self.kmeans_model.transform(data)
        min_distance = np.min(distances)
        
        if min_distance > 1.2:  # Slightly higher threshold for better precision
            self.kmeans_model.partial_fit(data)
            return self._create_novel_concept(data, domain, reward, "distant_cluster")
        
        return None, 0.0

    def _create_novel_concept(self, data: np.ndarray, domain: str, reward: float, 
                            discovery_type: str) -> Tuple[str, float]:
        """Create a new concept with enhanced metadata."""
        self.concept_counter += 1
        concept_name = f"HCT_Concept_{domain}_{self.concept_counter}"
        
        # Calculate concept properties
        concept_variance = np.var(data)
        concept_novelty = min(1.0, concept_variance * 2.0)  # Scale variance to 0-1 range
        
        concept_data = {
            "type": "discovered_concept",
            "domain": domain,
            "reward": reward,
            "novelty_score": concept_novelty,
            "discovery_type": discovery_type,
            "description": f"An emergent pattern discovered through {discovery_type} analysis.",
            "discovery_timestamp": datetime.now().isoformat(),
            "data_statistics": {
                "mean": float(np.mean(data)),
                "variance": float(concept_variance),
                "samples": data.shape[0]
            }
        }
        
        self.ckg.add_node(concept_name, concept_data)
        
        # Create relationship to domain
        relationship_name = f"HCT_Relationship_{self.relationship_types_counter}"
        self.ckg.propose_new_relationship_type(
            relationship_name, 
            f"An emergent relationship discovered by HCT through {discovery_type} analysis."
        )
        self.relationship_types_counter += 1
        self.ckg.add_edge(domain, concept_name, relationship_name)
        
        # Determine bonus based on novelty
        bonus = 3.0 + (concept_novelty * 2.0)  # Base bonus + novelty multiplier
        
        # Architectural innovation for highly novel concepts
        if concept_novelty > 0.8:
            self._propose_architectural_innovation(concept_name, concept_novelty)
        
        return concept_name, bonus

    def _analyze_cross_domain_patterns(self, data: np.ndarray, current_domain: str) -> float:
        """Analyze patterns across different domains for cross-domain insights."""
        cross_domain_bonus = 0.0
        
        # Get concepts from other domains
        other_domain_concepts = [
            node_id for node_id, node_data in self.ckg.db.nodes.items()
            if node_data.get('type') == 'discovered_concept' 
            and node_data.get('domain') != current_domain
        ]
        
        if not other_domain_concepts:
            return 0.0
        
        # Simple cross-domain similarity analysis (would be more sophisticated)
        for concept_id in other_domain_concepts:
            concept_data = self.ckg.query(concept_id)
            if 'data_statistics' in concept_data.get('node', {}):
                other_domain_mean = concept_data['node']['data_statistics'].get('mean', 0)
                current_mean = np.mean(data)
                
                # Bonus for similar patterns across domains
                similarity = 1.0 - abs(other_domain_mean - current_mean) / max(abs(other_domain_mean), abs(current_mean), 0.001)
                if similarity > 0.7:
                    cross_domain_bonus += similarity * 2.0
                    
                    # Record cross-domain insight
                    insight_id = f"cross_domain_insight_{len(self.cross_domain_insights)}"
                    self.cross_domain_insights[insight_id] = {
                        'domains': [current_domain, concept_data['node'].get('domain')],
                        'similarity': similarity,
                        'timestamp': datetime.now().isoformat()
                    }
        
        return cross_domain_bonus

    def _is_novel_pattern(self, representation: torch.Tensor, domain: str) -> bool:
        """Enhanced novelty detection with multiple criteria."""
        rep_np = representation.detach().cpu().numpy()
        
        # Multiple novelty indicators
        variance = np.var(rep_np)
        max_val = np.max(np.abs(rep_np))
        
        # Adaptive threshold based on domain experience
        domain_experience = sum(1 for n in self.ckg.db.nodes.values() 
                              if n.get('domain') == domain and n.get('type') == 'discovered_concept')
        novelty_threshold = 0.5 + (domain_experience * 0.1)  # Higher threshold for experienced domains
        
        return variance > novelty_threshold or max_val > 2.0

    def _propose_architectural_innovation(self, concept_name: str, novelty_score: float):
        """Propose architectural upgrades for highly novel concepts."""
        upgrade_type = self._determine_appropriate_upgrade(novelty_score)
        
        proposal_id = f"proposal_{self.ckg.proposal_counter}"
        proposal = {
            "proposal_id": proposal_id,
            "proposer": "HCT",
            "type": upgrade_type,
            "predicted_impact": self.arch_layer.predict_impact(upgrade_type),
            "proposed_changes": self.arch_layer.get_proposed_changes(upgrade_type),
            "novelty_score": novelty_score,
            "trigger_concept": concept_name,
            "status": "pending_human_review",
            "timestamp": datetime.now().isoformat()
        }
        
        self.ckg.add_node(proposal_id, proposal)
        self.ckg.add_edge("HCT", proposal_id, "PROPOSED_UPGRADE")
        self.ckg.add_edge(concept_name, proposal_id, "TRIGGERED_INNOVATION")
        
        print(f"\n[HCT] Proposed architectural innovation '{upgrade_type}' triggered by concept '{concept_name}'")

    def _determine_appropriate_upgrade(self, novelty_score: float) -> str:
        """Determine the most appropriate architectural upgrade based on novelty."""
        if novelty_score > 0.9:
            return "add_new_expert_with_specialization"
        elif novelty_score > 0.7:
            return "enhance_attention_mechanism"
        elif novelty_score > 0.5:
            return "add_conceptual_memory_slot"
        else:
            return "adjust_learning_parameters"

    def get_innovation_report(self) -> Dict:
        """Get a report on innovation and discovery activities."""
        return {
            "total_concepts_discovered": self.concept_counter,
            "cross_domain_insights": len(self.cross_domain_insights),
            "innovation_score": self.innovation_score,
            "active_clusters": self.num_clusters,
            "recent_innovations": list(self.cross_domain_insights.keys())[-5:] if self.cross_domain_insights else []
        }

    def adapt_discovery_parameters(self, success_rate: float):
        """Adapt discovery parameters based on success rate."""
        if success_rate > 0.7:
            # Increase aggression for successful discovery
            self.discovery_threshold *= 0.9
            self.num_clusters = min(15, self.num_clusters + 1)
        elif success_rate < 0.3:
            # Become more conservative
            self.discovery_threshold *= 1.1
            self.num_clusters = max(3, self.num_clusters - 1)