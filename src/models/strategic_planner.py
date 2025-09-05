# /src/models/strategic_planner.py

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph

class StrategicPlanner:
    """
    The Strategic Planner is responsible for setting high-level, long-term goals
    for the ARLC. It now selects goals dynamically from the CKG.
    """
    def __init__(self, model: nn.Module, ckg: ConceptualKnowledgeGraph):
        self.model = model
        self.ckg = ckg
        self.current_goal = None
        self._initialize_strategic_goals()

    def _initialize_strategic_goals(self):
        """Initializes strategic goals in the CKG if they don't exist."""
        goals = {
            'control_center': {'domain': 'chess', 'description': 'Maintain control over the central squares.'},
            'king_safety': {'domain': 'chess', 'description': 'Protect the king from threats.'},
            'material_advantage': {'domain': 'chess', 'description': 'Gain more valuable pieces than the opponent.'},
            'minimize_gaps': {'domain': 'tetris', 'description': 'Reduce the number of empty spaces on the board.'},
            'clear_lines': {'domain': 'tetris', 'description': 'Complete horizontal lines to clear them from the board.'},
            'HCT_Novelty_Score': {'domain': 'all', 'description': 'Prioritize moves that lead to newly discovered concepts.'}
        }
        for goal_id, props in goals.items():
            if not self.ckg.query(goal_id):
                self.ckg.add_node(goal_id, {"type": "strategic_goal", **props})
                print(f"Added strategic goal to CKG: {goal_id}")

    def select_goal(self, conceptual_features: torch.Tensor, domain: str) -> Dict[str, Any]:
        """
        Analyzes the conceptual features and selects the most relevant long-term goal
        by querying the CKG.
        """
        self._initialize_strategic_goals()

        # New: Retrieve relevant conceptual information from the CKG based on the domain.
        conceptual_info = self.ckg.query(f"{domain}_conceptual_state")
        
        # New: The logic for selecting a goal is now based on CKG knowledge,
        # which can include information from other modules like the Adversarial Module or HCT.
        
        if domain == 'chess':
            king_safety_node = self.ckg.query('king_safety')
            # The decision is based on a CKG query, not a hard-coded check.
            if king_safety_node and 'is_king_threatened' in king_safety_node['node'].get('properties', {}):
                self.current_goal = 'king_safety'
                return {'goal': 'king_safety', 'description': king_safety_node['node']['description']}
            
            material_node = self.ckg.query('material_advantage')
            if material_node and 'is_material_disadvantaged' in material_node['node'].get('properties', {}):
                self.current_goal = 'material_advantage'
                return {'goal': 'material_advantage', 'description': material_node['node']['description']}
                
            center_control_node = self.ckg.query('control_center')
            self.current_goal = 'control_center'
            return {'goal': 'control_center', 'description': center_control_node['node']['description']}

        elif domain == 'tetris':
            gaps_node = self.ckg.query('minimize_gaps')
            if gaps_node and 'has_many_gaps' in gaps_node['node'].get('properties', {}):
                self.current_goal = 'minimize_gaps'
                return {'goal': 'minimize_gaps', 'description': gaps_node['node']['description']}
                
            clear_lines_node = self.ckg.query('clear_lines')
            self.current_goal = 'clear_lines'
            return {'goal': 'clear_lines', 'description': clear_lines_node['node']['description']}

        # Check for HCT discovered goals
        hct_goals = [n for n in self.ckg.nodes.keys() if n.startswith('HCT_Concept')]
        if hct_goals:
            novelty_node = self.ckg.query('HCT_Novelty_Score')
            self.current_goal = 'HCT_Novelty_Score'
            return {'goal': 'HCT_Novelty_Score', 'description': novelty_node['node']['description']}

        self.current_goal = 'none'
        return {'goal': 'none', 'description': 'No specific strategic goal selected.'}
