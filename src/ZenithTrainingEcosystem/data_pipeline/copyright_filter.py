"""
Copyright Filter - Detects and handles potentially copyrighted content
"""

import re
import hashlib
from typing import List, Dict, Any

class CopyrightFilter:
    def __init__(self):
        self.copyright_indicators = [
            r"©\s*\d{4}",
            r"copyright\s*\d{4}",
            r"all rights reserved",
            r"patent\s*(?:no|number)?\s*[:\s]*[\w\d\-]+",
            r"ISBN\s*[:\s]*[\d\-X]+",
            r"ISSN\s*[:\s]*[\d\-X]+"
        ]
        self.known_copyrighted_patterns = self.load_known_patterns()
        
    def load_known_patterns(self) -> List[str]:
        """Load known copyrighted content patterns"""
        # This would connect to your Conceptual Knowledge Graph
        return [
            # Add patterns for known copyrighted characters, phrases, etc.
        ]
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text for potential copyright issues"""
        analysis = {
            'has_copyright_indicators': False,
            'copyright_confidence': 0.0,
            'flagged_sections': [],
            'recommendation': 'safe',  # safe, review, unsafe
            'content_hash': self._generate_content_hash(text)
        }
        
        # Check for copyright indicators
        indicators_found = []
        for pattern in self.copyright_indicators:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                indicators_found.append({
                    'pattern': pattern,
                    'match': match.group(),
                    'position': match.span()
                })
        
        if indicators_found:
            analysis['has_copyright_indicators'] = True
            analysis['copyright_confidence'] = min(0.7 + (len(indicators_found) * 0.1), 1.0)
            analysis['flagged_sections'] = indicators_found
            analysis['recommendation'] = 'review'
        
        # Check against known copyrighted patterns
        pattern_matches = self.check_known_patterns(text)
        if pattern_matches:
            analysis['copyright_confidence'] = max(analysis['copyright_confidence'], 0.9)
            analysis['flagged_sections'].extend(pattern_matches)
            analysis['recommendation'] = 'unsafe'
        
        return analysis
    
    def check_known_patterns(self, text: str) -> List[Dict]:
        """Check against database of known copyrighted content"""
        matches = []
        # This would query your external copyright database or CKG
        return matches
    
    def _generate_content_hash(self, text: str) -> str:
        """Generate hash for content tracking"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def filter_dataset(self, dataset: List[Dict]) -> List[Dict]:
        """Filter entire dataset for copyright issues"""
        filtered_data = []
        
        for item in dataset:
            if isinstance(item, str):
                text = item
                metadata = {}
            else:
                text = item.get('text', '')
                metadata = item
            
            analysis = self.analyze_text(text)
            
            if analysis['recommendation'] != 'unsafe':
                filtered_item = {
                    'text': text,
                    'metadata': metadata,
                    'copyright_analysis': analysis
                }
                filtered_data.append(filtered_item)
        
        return filtered_data
