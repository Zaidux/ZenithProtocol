# /src/nlp/command_parser.py

import spacy
from typing import Dict, Any

class CommandParser:
    """
    Parses natural language queries and extracts a structured command
    and its parameters.
    """
    def __init__(self):
        # We can use a simple, rule-based approach for this.
        # For a full implementation, a fine-tuned NLP model would be used.
        self.command_map = {
            "explain": ["why", "explain", "reasoning", "justification"],
            "strategy": ["strategy", "long-term", "plan", "goal"],
            "board_state": ["board", "state", "situation"],
            "eval_move": ["evaluate", "score", "value", "assess"]
        }

    def parse_command(self, query: str) -> Dict[str, Any]:
        """
        Takes a natural language query and returns a structured command.
        
        Args:
            query (str): The user's input string.
            
        Returns:
            Dict: A dictionary containing the command and any relevant parameters.
        """
        lower_query = query.lower()
        command = "unknown"
        for cmd, keywords in self.command_map.items():
            if any(k in lower_query for k in keywords):
                command = cmd
                break
        
        # Simple entity extraction (e.g., for a specific move)
        # For a more advanced version, we would use a library like spaCy.
        entities = {}
        tokens = lower_query.split()
        if "move" in tokens:
            try:
                move_idx = tokens[tokens.index("move") + 1]
                entities['move'] = int(move_idx)
            except (ValueError, IndexError):
                pass
        
        return {
            "command": command,
            "entities": entities
        }

