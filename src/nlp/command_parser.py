# src/nlp/command_parser.py

"""
Enhanced Command Parser with Contextual Understanding
====================================================
Now supports counterfactual queries and better entity recognition.
"""

import spacy
from typing import Dict, Any, List, Optional
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph
import re

class CommandParser:
    """
    Enhanced parser with counterfactual understanding and context awareness.
    """
    def __init__(self, ckg: ConceptualKnowledgeGraph):
        self.ckg = ckg
        self.command_map = {
            "explain": ["why", "explain", "justify", "reason"],
            "counterfactual": ["what if", "why not", "instead of", "consider"],
            "strategy": ["plan", "strategy", "approach", "method"],
            "eval_move": ["evaluate", "assess", "score", "value"],
            "what_if": ["simulate", "hypothetical", "suppose"],
            "organize_files": ["organize", "arrange", "sort", "categorize"]
        }
        
        try:
            self.nlp = spacy.load("en_core_web_md")  # Medium model for better accuracy
        except:
            print("Spacy model not found. Using basic parsing.")
            self.nlp = None

        self._initialize_advanced_parser()

    def _initialize_advanced_parser(self):
        """Initialize with advanced concept relationships."""
        # Add counterfactual concepts
        counterfactual_concepts = {
            "alternative": ["different", "other", "another"],
            "rejection": ["reject", "dismiss", "ignore"],
            "comparison": ["compare", "versus", "against"]
        }
        
        for concept, words in counterfactual_concepts.items():
            for word in words:
                self.ckg.add_node(word, {"type": "counterfactual_concept", "concept": concept})

    def parse_command(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Enhanced parsing with counterfactual detection and context awareness.
        """
        if not self.nlp:
            return self._basic_parse(query)

        doc = self.nlp(query.lower())
        command = "unknown"
        entities = {}
        is_counterfactual = False

        # Detect counterfactual queries
        if any(phrase in query.lower() for phrase in ["why didn't", "why did you not", "what about"]):
            command = "counterfactual"
            is_counterfactual = True
            entities["type"] = "alternative_rejection"

        # Extract main command
        if not is_counterfactual:
            for token in doc:
                node_info = self.ckg.query(token.text)
                if node_info and node_info['node'].get('type') == 'command_keyword':
                    command = node_info['node'].get('command')
                    break

        # Enhanced entity extraction
        entities.update(self._extract_entities(doc, command, context))
        
        # Extract counterfactual alternatives
        if is_counterfactual:
            entities["alternative"] = self._extract_alternative(doc)

        return {
            "command": command,
            "entities": entities,
            "is_counterfactual": is_counterfactual,
            "original_query": query
        }

    def _extract_alternative(self, doc) -> str:
        """Extract proposed alternative from counterfactual query."""
        # Look for patterns like "why didn't you [X] instead"
        for i, token in enumerate(doc):
            if token.text in ["instead", "rather", "alternative"]:
                # Extract the alternative phrase
                start_idx = max(0, i - 3)
                end_idx = min(len(doc), i + 3)
                alternative = " ".join(token.text for token in doc[start_idx:end_idx])
                return alternative
        
        return "unknown_alternative"

    def _extract_entities(self, doc, command: str, context: Optional[Dict]) -> Dict:
        """Enhanced entity extraction with context awareness."""
        entities = {}

        if command in ["eval_move", "what_if"]:
            # Extract move numbers or coordinates
            for token in doc:
                if token.like_num:
                    entities['move'] = int(token.text)
                elif len(token.text) == 2 and token.text[0].isalpha() and token.text[1].isdigit():
                    entities['move'] = token.text

        elif command == "organize_files":
            # Enhanced file organization parsing
            entities = self._parse_organization_query(doc, context)

        elif command == "counterfactual":
            # Extract what alternative is being suggested
            entities["suggested_alternative"] = self._find_suggested_alternative(doc)

        return entities

    def _parse_organization_query(self, doc, context: Optional[Dict]) -> Dict:
        """Parse file organization queries with context awareness."""
        entities = {
            "folder_name": None,
            "file_types": [],
            "categories": [],
            "priority": "medium"
        }

        for i, token in enumerate(doc):
            # Detect file types
            if token.text in ["photo", "photos", "image", "images"]:
                entities["file_types"].extend(['.jpg', '.png', '.jpeg'])
            elif token.text in ["document", "documents"]:
                entities["file_types"].extend(['.doc', '.pdf', '.txt'])
            
            # Detect organization categories
            if token.text in ["date", "time", "chronological"]:
                entities["categories"].append("date")
            elif token.text in ["location", "place", "where"]:
                entities["categories"].append("location")
            elif token.text in ["event", "activity"]:
                entities["categories"].append("event")
            
            # Detect priority
            if token.text in ["urgent", "important", "priority"]:
                entities["priority"] = "high"

        # Use context if available
        if context and "default_categories" in context:
            entities["categories"].extend(context["default_categories"])

        return entities

    def _find_suggested_alternative(self, doc) -> str:
        """Find the alternative being suggested in counterfactual queries."""
        # Look for patterns suggesting alternatives
        patterns = [
            r"why didn't you ([\w\s]+) instead",
            r"what about ([\w\s]+)",
            r"why not ([\w\s]+)"
        ]
        
        query_text = " ".join(token.text for token in doc)
        
        for pattern in patterns:
            match = re.search(pattern, query_text)
            if match:
                return match.group(1).strip()
        
        return "unknown_alternative"