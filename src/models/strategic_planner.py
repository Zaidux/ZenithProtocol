# /src/models/strategic_planner.py

import torch
import torch.nn as nn
from typing import Dict, Any

class StrategicPlanner:
    """
    The Strategic Planner is responsible for setting high-level, long-term goals
    for the ARLC to pursue.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        # A dictionary of predefined, high-level strategic goals for each domain.
        # These will be matched to the conceptual features identified by the HCT layer.
        self.strategic_goals = {
            'chess': {
                'control_center': ['White Center Control', 'Black Center Control'],
                'king_safety': ['White King Safety', 'Black King Safety'],
                'material_advantage': ['Material Advantage'],
                'explore_novelty': ['HCT_Novelty_Score'] # A placeholder for a concept discovered by HCT
            },
            'tetris': {
                'minimize_gaps': ['Gaps'],
                'clear_lines': ['Lines Cleared'],
                'keep_low_profile': ['Max Height']
            }
        }
        self.current_goal = None

    def select_goal(self, conceptual_features: torch.Tensor, domain: str) -> Dict[str, Any]:
        """
        Analyzes the conceptual features and selects the most relevant long-term goal.
        This is a simplified, rule-based selection. An advanced version would use
        a separate neural network trained for goal selection.
        """
        # For simplicity, we'll choose the goal that needs the most attention.
        # In Chess, if King safety is low, that becomes a top priority.
        if domain == 'chess':
            # Check for immediate threats to king safety
            king_safety_idx = self.model.hct_layer.conceptual_feature_names['chess'].index('White King Safety')
            if conceptual_features[0, king_safety_idx] < 0.2: # Example threshold
                self.current_goal = 'king_safety'
                return {'goal': 'king_safety', 'description': 'The model is prioritizing king safety due to a low conceptual score.'}
            
            # If no immediate threat, focus on material or center control
            material_advantage_idx = self.model.hct_layer.conceptual_feature_names['chess'].index('Material Advantage')
            if conceptual_features[0, material_advantage_idx] < 0:
                self.current_goal = 'material_advantage'
                return {'goal': 'material_advantage', 'description': 'The model is trying to regain material advantage.'}
                
            self.current_goal = 'control_center'
            return {'goal': 'control_center', 'description': 'The model is focusing on controlling the board center.'}

        # For Tetris, a simpler set of rules
        elif domain == 'tetris':
            gaps_idx = self.model.hct_layer.conceptual_feature_names['tetris'].index('Gaps')
            if conceptual_features[0, gaps_idx] > 0.5:
                self.current_goal = 'minimize_gaps'
                return {'goal': 'minimize_gaps', 'description': 'The model is prioritizing minimizing gaps.'}
            
            self.current_goal = 'clear_lines'
            return {'goal': 'clear_lines', 'description': 'The model is focusing on clearing lines.'}

        return {'goal': 'none', 'description': 'No specific strategic goal selected.'}
