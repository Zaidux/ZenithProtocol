# src/conceptual_knowledge_graph/ckg.py

"""
Enhanced Conceptual Knowledge Graph with Causal Rules and Tracing Methods
=======================================================================
Adds support for causal reasoning, rule-based validation, and traceable explanations.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
import hashlib
import numpy as np
import torch

from ..conceptual_encoder.conceptual_encoder import ZenithConceptualEncoder
from .in_memory_db import InMemoryGraphDB

try:
    import blockchain_interface_cpp
except ImportError:
    blockchain_interface_cpp = None
    print("Warning: Blockchain interface not found. Running in local-only mode.")

class ConceptualKnowledgeGraph:
    """
    An in-memory, graph-based knowledge store for the Zenith Protocol.
    Now enhanced with causal rules, tracing methods, and rule-based validation.
    """
    def __init__(self, storage_path: str = "conceptual_graph.json"):
        self.db = InMemoryGraphDB()
        self.storage_path = storage_path
        self.proposal_counter = 0
        self.causal_rules = {}  # New: Store causal rules separately for efficient access
        self.reward_rules = {}   # New: Store reward calculation rules

        self.blockchain_enabled = blockchain_interface_cpp is not None
        if self.blockchain_enabled:
            self.blockchain_client = blockchain_interface_cpp.BlockchainInterface()

        self.relationship_types = self._get_default_relationship_types()
        self._load_graph()
        self._initialize_causal_rule_system()  # New: Initialize causal rule system

    def _get_default_relationship_types(self) -> Dict[str, Dict[str, Any]]:
        """Enhanced with causal relationship types."""
        base_relationships = {
            "IS_A": {"description": "A conceptual inheritance."},
            "HAS_PROPERTY": {"description": "Links a concept to its property."},
            "PERFORMS": {"description": "Links an Agent to an Action."},
            "ACTS_ON": {"description": "Links an Action to an Object."},
            "HAS_REASON": {"description": "Links an Action to its Reason."},
            "HAS_DISCOVERED_CONCEPT": {"description": "Links a domain to a discovered concept."},
            "IS_VISUAL": {"description": "Links a concept to a visual element."},
            "IS_AUDIO": {"description": "Links a concept to an audio element."},
            "HAS_TONE": {"description": "Links a concept to a socio-linguistic tone."},
            "PROPOSED_UPGRADE": {"description": "Links a component to an architectural upgrade proposal."},
            "FINALIZED": {"description": "Links a human decision to an architectural proposal."},
            # New causal relationship types
            "CAUSES": {"description": "Indicates a causal relationship between concepts."},
            "PREVENTS": {"description": "Indicates that one concept prevents another."},
            "ENABLES": {"description": "Indicates that one concept enables another."},
            "REQUIRES": {"description": "Indicates a prerequisite relationship."},
            "LEADS_TO": {"description": "Indicates a sequential or consequential relationship."},
        }
        return base_relationships

    def _initialize_causal_rule_system(self):
        """Initialize the causal rule system with domain-specific rules."""
        # Tetris-specific causal rules
        self._add_domain_rules('tetris', [
            {
                'id': 'rule_line_clear',
                'description': 'Completing a row clears the line and scores points',
                'conditions': ['row_completion=100%'],
                'effects': ['lines_cleared+=1', 'score+=100'],
                'confidence': 0.95,
                'domain': 'tetris'
            },
            {
                'id': 'rule_gap_creation',
                'description': 'Placing certain blocks in certain positions creates gaps',
                'conditions': ['block_type in [S, Z]', 'position_above_ledge=true'],
                'effects': ['gaps+=1', 'future_risk+=0.3'],
                'confidence': 0.85,
                'domain': 'tetris'
            },
            {
                'id': 'rule_height_penalty',
                'description': 'Higher stack height increases risk',
                'conditions': ['max_height>15'],
                'effects': ['risk_multiplier=1.5'],
                'confidence': 0.9,
                'domain': 'tetris'
            }
        ])

        # Chess-specific causal rules
        self._add_domain_rules('chess', [
            {
                'id': 'rule_material_advantage',
                'description': 'Having more pieces provides strategic advantage',
                'conditions': ['material_balance>0'],
                'effects': ['strategic_advantage+=material_balance*0.1'],
                'confidence': 0.88,
                'domain': 'chess'
            }
        ])

        # General reward rules
        self._initialize_reward_rules()

    def _add_domain_rules(self, domain: str, rules: List[Dict]):
        """Add multiple rules for a specific domain."""
        for rule in rules:
            self.add_causal_rule(rule['id'], rule)

    def _initialize_reward_rules(self):
        """Initialize rules for reward calculation."""
        self.reward_rules = {
            'tetris': {
                'lines_cleared': {'weight': 10.0, 'description': 'Points per line cleared'},
                'gaps': {'weight': -2.0, 'description': 'Penalty per gap created'},
                'max_height': {'weight': -1.0, 'description': 'Penalty for high stack'},
                'board_fullness': {'weight': -0.5, 'description': 'Penalty for crowded board'}
            },
            'chess': {
                'material_advantage': {'weight': 5.0, 'description': 'Reward for material advantage'},
                'king_safety': {'weight': 3.0, 'description': 'Reward for king safety'},
                'center_control': {'weight': 2.0, 'description': 'Reward for center control'},
                'development': {'weight': 1.5, 'description': 'Reward for piece development'}
            }
        }

    def _load_graph(self):
        """Load graph data from storage, including causal rules."""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.db.nodes = data.get("nodes", {})
                self.db.edges = data.get("edges", {})
                self.relationship_types = data.get("relationship_types", self._get_default_relationship_types())
                # Load causal rules if they exist
                self.causal_rules = data.get("causal_rules", {})
                self.reward_rules = data.get("reward_rules", {})
            print(f"Conceptual Knowledge Graph loaded from {self.storage_path}")
        except (FileNotFoundError, json.JSONDecodeError):
            print("No existing graph found. Starting with a new one.")
            self._initialize_base_concepts()
            self._save_graph()

    def _save_graph(self):
        """Save graph data including causal rules."""
        data = {
            "nodes": self.db.nodes,
            "edges": self.db.edges,
            "relationship_types": self.relationship_types,
            "causal_rules": self.causal_rules,
            "reward_rules": self.reward_rules
        }

        if self.blockchain_enabled:
            data_to_store = json.dumps(data, sort_keys=True)
            self.blockchain_client.add_to_blockchain(data_to_store)
            print("Graph state hashed and stored on the mock blockchain.")

        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Conceptual Knowledge Graph saved to {self.storage_path}")

    def _initialize_base_concepts(self):
        base_concepts = {
            "Agent": {"type": "concept", "description": "Entity that performs actions."},
            "Action": {"type": "concept", "description": "A verb or process executed."},
            "Object": {"type": "concept", "description": "A tangible item with properties."},
            "Reason": {"type": "concept", "description": "The 'why' behind an action."},
            "Internet": {"type": "concept", "description": "A global network of computers."},
            "Real-Time Data": {"type": "concept", "description": "Information updated constantly."},
            "Visual_Data": {"type": "modality", "description": "Represents information from an image or video."},
            "Audio_Data": {"type": "modality", "description": "Represents information from an audio clip."},
            "Socio-Linguistic_Context": {"type": "modality", "description": "Represents conversational tone and style."},
        }
        for node_id, data in base_concepts.items():
            self.add_node(node_id, {**data, "verifiability_score": 1.0})

    def add_node(self, node_id: str, properties: Dict[str, Any]) -> None:
        self.db.add_node(node_id, properties)
        if node_id.startswith("proposal_"):
            self.proposal_counter += 1
        self._save_graph()

    def add_edge(self, source_id: str, target_id: str, relationship: str, properties: Optional[Dict[str, Any]] = None) -> None:
        if relationship not in self.relationship_types:
            print(f"Warning: Relationship type '{relationship}' is not in the schema. Adding a default entry.")
            self.propose_new_relationship_type(relationship, "Dynamically created relationship.")

        self.db.add_edge(source_id, target_id, relationship, properties)
        self._save_graph()

    def update_node_properties(self, node_id: str, new_properties: Dict[str, Any]) -> None:
        self.db.update_node_properties(node_id, new_properties)
        self._save_graph()

    def query(self, entity_id: str) -> Optional[Dict[str, Any]]:
        return self.db.query(entity_id)

    def query_by_property(self, key: str, value: Any) -> List[Dict[str, Any]]:
        """
        Queries the graph for all nodes that match a specific key-value pair in their properties.
        This is crucial for finding all pending architectural proposals.
        """
        matching_nodes = []
        for node_id, node_data in self.db.nodes.items():
            if node_data.get(key) == value:
                # Return the full node data with its connections.
                matching_nodes.append(self.query(node_id))
        return matching_nodes

    def add_prompt_response(self, prompt: str, response: str, conceptual_encoder: 'ZenithConceptualEncoder') -> None:
        extracted_data = conceptual_encoder.extract_concepts_and_relations(prompt, response)

        for data in extracted_data:
            source_id = data.get("source_id")
            source_type = data.get("source_type")
            target_id = data.get("target_id")
            target_type = data.get("target_type")
            relationship = data.get("relationship")
            properties = data.get("properties", {})

            if not self.db.node_exists(source_id):
                self.add_node(source_id, {"type": source_type, "content": source_id, "verifiability_score": 0.5})
            if not self.db.node_exists(target_id):
                self.add_node(target_id, {"type": target_type, "content": target_id, "verifiability_score": 0.5})

            self.add_edge(source_id, target_id, relationship, properties)
            self.db.update_verifiability_score(source_id, 0.1)
            self.db.update_verifiability_score(target_id, 0.1)

        self._save_graph()

    def propose_new_relationship_type(self, relationship_name: str, description: str, is_vulnerability: bool = False):
        if relationship_name not in self.relationship_types:
            self.relationship_types[relationship_name] = {
                "description": description,
                "is_vulnerability": is_vulnerability,
                "created_by": "HCT" if not is_vulnerability else "AdversarialModule",
                "timestamp": datetime.now().isoformat()
            }
            print(f"Proposed new relationship type: '{relationship_name}'")
            self._save_graph()

    def get_verifiability_score(self, node_id: str) -> float:
        node = self.db.query(node_id)
        if node and "node" in node and "verifiability_score" in node["node"]:
            return node["node"]["verifiability_score"]
        return 0.0

    def get_verifiable_record(self, data_id: str) -> Optional[Dict[str, Any]]:
        if self.blockchain_enabled:
            record_hash = self.blockchain_client.generate_hash(data_id)
            tx = self.blockchain_client.get_from_blockchain(record_hash)

            if tx.data_hash:
                local_record = self.query(data_id)
                local_data_string = json.dumps(local_record, sort_keys=True)
                if self.blockchain_client.verify_transaction(tx, local_data_string):
                    print(f"Verifiable record for '{data_id}' found on blockchain.")
                    return {"local_data": local_record, "blockchain_record": tx}
                else:
                    print(f"Warning: Local data for '{data_id}' does not match blockchain record.")
                    return None
        return None

    def add_causal_rule(self, rule_id: str, rule_data: Dict[str, Any]):
        """
        Add a new causal rule to the knowledge graph.
        
        Args:
            rule_id: Unique identifier for the rule
            rule_data: Rule definition including conditions, effects, etc.
        """
        self.causal_rules[rule_id] = {
            **rule_data,
            'created_at': datetime.now().isoformat(),
            'last_used': None,
            'usage_count': 0
        }
        self._save_graph()
        print(f"Added causal rule: {rule_id} - {rule_data.get('description', 'No description')}")

    def get_reward_rule(self, concept: str, domain: str) -> Optional[Dict]:
        """
        Get the reward rule for a specific concept in a domain.
        
        Args:
            concept: The conceptual feature (e.g., 'lines_cleared')
            domain: The problem domain ('tetris', 'chess', etc.)
            
        Returns:
            Reward rule dictionary or None if not found
        """
        domain_rules = self.reward_rules.get(domain, {})
        return domain_rules.get(concept)

    def validate_forecast(self, conceptual_features: torch.Tensor, action: Any, 
                         domain: str) -> Dict[str, Any]:
        """
        Validate a forecasted state against causal rules in the CKG.
        
        Args:
            conceptual_features: Forecasted conceptual features
            action: The action that led to this forecast
            domain: The problem domain
            
        Returns:
            Validation result with applied rules, violated rules, and confidence
        """
        validation_result = {
            'is_valid': True,
            'applied_rules': [],
            'violated_rules': [],
            'confidence': 1.0,
            'warnings': []
        }

        # Convert tensor to dictionary for rule checking (simplified example)
        # In practice, this would use a more sophisticated feature extraction
        features_dict = self._extract_features_dict(conceptual_features, domain)
        
        # Check all rules for this domain
        for rule_id, rule in self.causal_rules.items():
            if rule.get('domain') == domain or rule.get('domain') is None:
                rule_applies = self._check_rule_conditions(rule, features_dict, action)
                
                if rule_applies:
                    validation_result['applied_rules'].append(rule_id)
                    # Update confidence based on rule confidence
                    validation_result['confidence'] *= rule.get('confidence', 0.8)
                    
                    # Check if rule effects are desirable
                    if not self._are_effects_desirable(rule, domain):
                        validation_result['is_valid'] = False
                        validation_result['violated_rules'].append(rule_id)
                        validation_result['warnings'].append(
                            f"Rule {rule_id} applied but with undesirable effects"
                        )
        
        # Update rule usage statistics
        for rule_id in validation_result['applied_rules']:
            self.causal_rules[rule_id]['usage_count'] += 1
            self.causal_rules[rule_id]['last_used'] = datetime.now().isoformat()

        return validation_result

    def _extract_features_dict(self, conceptual_features: torch.Tensor, domain: str) -> Dict:
        """
        Extract conceptual features as a dictionary for rule checking.
        This is a simplified example - in practice, would use proper feature mapping.
        """
        features_np = conceptual_features.detach().cpu().numpy()
        if domain == 'tetris':
            return {
                'lines_cleared': float(features_np[0][0] if features_np.size > 0 else 0),
                'gaps': float(features_np[0][1] if features_np.size > 1 else 0),
                'max_height': float(features_np[0][2] if features_np.size > 2 else 0),
                'board_fullness': float(features_np[0][3] if features_np.size > 3 else 0)
            }
        elif domain == 'chess':
            return {
                'material_advantage': float(features_np[0][0] if features_np.size > 0 else 0),
                'king_safety': float(features_np[0][1] if features_np.size > 1 else 0),
                'center_control': float(features_np[0][2] if features_np.size > 2 else 0)
            }
        return {}

    def _check_rule_conditions(self, rule: Dict, features: Dict, action: Any) -> bool:
        """
        Check if a rule's conditions are satisfied.
        Simplified implementation - would use a proper condition parser in practice.
        """
        conditions = rule.get('conditions', [])
        for condition in conditions:
            # Simple condition checking - in practice, use a proper expression evaluator
            if '=' in condition:
                var, value = condition.split('=', 1)
                var = var.strip()
                value = value.strip()
                
                current_value = features.get(var, 0)
                try:
                    if not self._compare_values(current_value, float(value), '='):
                        return False
                except ValueError:
                    # Handle non-numeric comparisons
                    if str(current_value) != value:
                        return False
        return True

    def _compare_values(self, actual, expected, operator: str) -> bool:
        """Compare values based on the given operator."""
        if operator == '=':
            return abs(actual - expected) < 0.001
        elif operator == '>':
            return actual > expected
        elif operator == '<':
            return actual < expected
        elif operator == '>=':
            return actual >= expected
        elif operator == '<=':
            return actual <= expected
        return False

    def _are_effects_desirable(self, rule: Dict, domain: str) -> bool:
        """
        Check if a rule's effects are generally desirable for the domain.
        """
        effects = rule.get('effects', [])
        for effect in effects:
            if any(bad_effect in effect for bad_effect in ['risk+=', 'penalty+=', 'gaps+=', 'danger+=']):
                return False
        return True

    def trace_causal_chain(self, action: str, outcome: str, max_depth: int = 5) -> Optional[Dict]:
        """
        Trace the causal chain from an action to an outcome.
        
        Args:
            action: The starting action
            outcome: The resulting outcome
            max_depth: Maximum depth to search
            
        Returns:
            Causal chain with intermediate steps and rules used
        """
        visited = set()
        chain = self._find_causal_path(action, outcome, visited, max_depth)
        
        if chain:
            return {
                'action': action,
                'outcome': outcome,
                'path': chain,
                'rules_used': self._extract_rules_from_path(chain),
                'confidence': self._calculate_chain_confidence(chain)
            }
        return None

    def _find_causal_path(self, current: str, target: str, visited: set, max_depth: int, 
                         current_path: List = None) -> Optional[List]:
        """Recursively find a causal path between two nodes."""
        if current_path is None:
            current_path = []
        
        if current == target:
            return current_path + [current]
        
        if len(current_path) >= max_depth or current in visited:
            return None
        
        visited.add(current)
        connections = self.db.query(current)
        
        if not connections:
            return None
        
        for connection in connections.get('connections', []):
            if connection['relationship'] in ['CAUSES', 'LEADS_TO', 'ENABLES']:
                next_node = connection['target_id']
                new_path = self._find_causal_path(
                    next_node, target, visited, max_depth, current_path + [current]
                )
                if new_path:
                    return new_path
        
        return None

    def _extract_rules_from_path(self, path: List[str]) -> List[str]:
        """Extract rules that explain the causal path."""
        rules_used = []
        for i in range(len(path) - 1):
            source, target = path[i], path[i+1]
            # Look for rules that connect these concepts
            for rule_id, rule in self.causal_rules.items():
                if self._rule_connects_concepts(rule, source, target):
                    rules_used.append(rule_id)
        return rules_used

    def _rule_connects_concepts(self, rule: Dict, source: str, target: str) -> bool:
        """Check if a rule connects two specific concepts."""
        # Simplified - in practice, would analyze rule conditions and effects
        description = rule.get('description', '').lower()
        return source.lower() in description and target.lower() in description

    def _calculate_chain_confidence(self, path: List[str]) -> float:
        """Calculate overall confidence for a causal chain."""
        if not path:
            return 0.0
        
        confidence = 1.0
        for i in range(len(path) - 1):
            # Look up connection confidence (simplified)
            connections = self.db.query(path[i])
            for conn in connections.get('connections', []):
                if conn['target_id'] == path[i+1]:
                    edge_confidence = conn.get('properties', {}).get('confidence', 0.7)
                    confidence *= edge_confidence
        return confidence

    def get_causal_explanation(self, action: str, outcome: str) -> str:
        """
        Generate a human-readable causal explanation.
        
        Args:
            action: The action taken
            outcome: The observed outcome
            
        Returns:
            Natural language explanation of the causal relationship
        """
        chain = self.trace_causal_chain(action, outcome)
        if not chain:
            return f"No clear causal chain found from {action} to {outcome}."
        
        explanation = f"The action '{action}' led to '{outcome}' through this causal chain:\n"
        
        for i, step in enumerate(chain['path']):
            if i < len(chain['path']) - 1:
                next_step = chain['path'][i+1]
                # Find the rule that explains this step
                rule = self._find_rule_for_transition(step, next_step)
                if rule:
                    explanation += f"  {i+1}. {step} → {next_step} (because: {rule['description']})\n"
                else:
                    explanation += f"  {i+1}. {step} → {next_step}\n"
        
        explanation += f"\nOverall confidence: {chain['confidence']:.2f}"
        return explanation

    def _find_rule_for_transition(self, source: str, target: str) -> Optional[Dict]:
        """Find a rule that explains the transition between two concepts."""
        for rule_id, rule in self.causal_rules.items():
            if self._rule_explains_transition(rule, source, target):
                return rule
        return None

    def _rule_explains_transition(self, rule: Dict, source: str, target: str) -> bool:
        """Check if a rule explains the transition between two concepts."""
        # Check if rule conditions mention source and effects mention target
        conditions = ' '.join(rule.get('conditions', [])).lower()
        effects = ' '.join(rule.get('effects', [])).lower()
        return source.lower() in conditions and target.lower() in effects

# --- Mock Classes for Demonstration ---
class ConceptualEncoder:
    def extract_concepts_and_relations(self, prompt: str, response: str) -> List[Dict[str, Any]]:
        return [
            {"source_id": "run", "source_type": "concept", "target_id": "Action", "target_type": "concept", "relationship": "IS_A", "properties": {"motion_type": "continuous"}},
        ]

class InMemoryGraphDB:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, node_id: str, properties: Dict[str, Any]) -> None:
        if node_id in self.nodes and self.nodes[node_id].get("type") != properties.get("type"):
            return
        self.nodes[node_id] = {
            **self.nodes.get(node_id, {}),
            **properties,
            "last_updated": datetime.now().isoformat()
        }

    def add_edge(self, source_id: str, target_id: str, relationship: str, properties: Optional[Dict[str, Any]] = None) -> None:
        if source_id not in self.nodes or target_id not in self.nodes:
            return
        edge_id = f"{source_id}_{relationship}_{target_id}"
        self.edges[edge_id] = {
            "source": source_id,
            "target": target_id,
            "relationship": relationship,
            "properties": properties or {},
            "created_at": datetime.now().isoformat()
        }

    def update_node_properties(self, node_id: str, new_properties: Dict[str, Any]) -> None:
        if node_id in self.nodes:
            self.nodes[node_id].update(new_properties)
            self.nodes[node_id]["last_updated"] = datetime.now().isoformat()
        else:
            pass

    def query(self, entity_id: str) -> Optional[Dict[str, Any]]:
        if entity_id not in self.nodes:
            return None
        node_data = self.nodes.get(entity_id, {})
        related_nodes = []
        for edge_id, edge in self.edges.items():
            if edge["source"] == entity_id:
                related_nodes.append({
                    "relationship": edge["relationship"],
                    "target_id": edge["target"],
                    "target_data": self.nodes.get(edge["target"]),
                    "edge_properties": edge["properties"]
                })
            elif edge["target"] == entity_id:
                related_nodes.append({
                    "relationship": edge["relationship"],
                    "source_id": edge["source"],
                    "source_data": self.nodes.get(edge["source"]),
                    "edge_properties": edge["properties"]
                })
        return {"node": node_data, "connections": related_nodes}

    def node_exists(self, node_id: str) -> bool:
        return node_id in self.nodes

    def update_verifiability_score(self, node_id: str, score_change: float) -> None:
        if node_id in self.nodes and "verifiability_score" in self.nodes[node_id]:
            current_score = self.nodes[node_id]["verifiability_score"]
            new_score = max(0.0, min(1.0, current_score + score_change))
            self.nodes[node_id]["verifiability_score"] = new_score
            self.nodes[node_id]["last_updated"] = datetime.now().isoformat()

# Example usage
if __name__ == '__main__':
    ckg = ConceptualKnowledgeGraph()
    conceptual_encoder = ConceptualEncoder()

    # Add a conversation
    prompt_text = "What is a good way to stay fit?"
    response_text = "Running is a great option because it helps burn calories."
    ckg.add_prompt_response(prompt_text, response_text, conceptual_encoder)

    # Add a multimodal concept
    ckg.add_node("red_car", {"type": "visual_concept", "modality": "visual", "properties": {"color": "red"}})
    ckg.add_node("high_pitch", {"type": "audio_concept", "modality": "audio", "properties": {"pitch": "high"}})
    ckg.add_edge("red_car", "Visual_Data", "IS_VISUAL")
    ckg.add_edge("high_pitch", "Audio_Data", "IS_AUDIO")
    ckg.add_edge("Running", "Socio-Linguistic_Context", "HAS_TONE", {"value": "positive"})

    # Test new causal reasoning capabilities
    print("Testing causal rule system...")
    mock_features = torch.tensor([[2.0, 1.0, 12.0, 0.3]])  # lines_cleared, gaps, max_height, board_fullness
    validation = ckg.validate_forecast(mock_features, 'place_s_block', 'tetris')
    print(f"Validation result: {validation}")
    
    # Test causal chain tracing
    print("\nTesting causal chain tracing...")
    chain = ckg.trace_causal_chain('place_s_block', 'gap_created')
    print(f"Causal chain: {chain}")
    
    # Test causal explanation
    print("\nTesting causal explanation...")
    explanation = ckg.get_causal_explanation('place_s_block', 'gap_created')
    print(f"Explanation:\n{explanation}")

    # Query the graph
    run_info = ckg.query("run")
    print("\n--- Querying 'run' ---")
    print(json.dumps(run_info, indent=4))