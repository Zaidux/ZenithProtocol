"""
Speculative Labeler - Identifies and labels speculative content with appropriate warnings
"""

import re
from typing import Dict, List, Any

class SpeculativeLabeler:
    def __init__(self):
        self.speculative_indicators = [
            r'\bmay\b', r'\bmight\b', r'\bcould\b', r'\bpossibly\b', r'\bperhaps\b',
            r'\bpotentially\b', r'\bsuggests\b', r'\bimplies\b', r'\bseems\b',
            r'\bappears\b', r'\blikely\b', r'\bunlikely\b', r'\bprobably\b',
            r'\bI think\b', r'\bI believe\b', r'\bin my opinion\b',
            r'\bsome\s+(?:people|scientists|experts)\b',
            r'\bit is thought\b', r'\bit is believed\b'
        ]
        
        self.speculative_patterns = [
            (r'if.*then', 0.7),
            (r'what if', 0.8),
            (r'suppose that', 0.6),
            (r'assuming that', 0.7),
            (r'in the future', 0.5),
            (r'one day', 0.4)
        ]
    
    def analyze_speculation(self, text: str) -> Dict[str, Any]:
        """Analyze text for speculative content and assign confidence"""
        analysis = {
            'is_speculative': False,
            'speculation_confidence': 0.0,
            'speculative_indicators': [],
            'speculation_type': 'none',  # hypothetical, predictive, opinion, none
            'recommended_warning': None
        }
        
        # Check for speculative language patterns
        indicators_found = []
        for pattern in self.speculative_indicators:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                indicators_found.append({
                    'indicator': match.group(),
                    'pattern': pattern,
                    'position': match.span()
                })
        
        # Check for complex speculative patterns
        pattern_matches = []
        for pattern, confidence in self.speculative_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                pattern_matches.append({
                    'pattern': pattern,
                    'base_confidence': confidence
                })
        
        # Calculate overall speculation confidence
        base_confidence = min(len(indicators_found) * 0.15, 0.8)
        pattern_confidence = max([p['base_confidence'] for p in pattern_matches] or [0])
        
        analysis['speculation_confidence'] = max(base_confidence, pattern_confidence)
        analysis['is_speculative'] = analysis['speculation_confidence'] > 0.3
        analysis['speculative_indicators'] = indicators_found + pattern_matches
        
        # Determine speculation type
        analysis['speculation_type'] = self.determine_speculation_type(
            text, indicators_found, pattern_matches
        )
        
        # Generate appropriate warning
        if analysis['is_speculative']:
            analysis['recommended_warning'] = self.generate_warning(
                analysis['speculation_type'],
                analysis['speculation_confidence']
            )
        
        return analysis
    
    def determine_speculation_type(self, text: str, indicators: List, patterns: List) -> str:
        """Determine the type of speculation"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['if', 'suppose', 'assuming']):
            return 'hypothetical'
        elif any(word in text_lower for word in ['will', 'going to', 'in the future']):
            return 'predictive'
        elif any(word in text_lower for word in ['think', 'believe', 'opinion']):
            return 'opinion'
        elif any(word in text_lower for word in ['may', 'might', 'could']):
            return 'possibility'
        else:
            return 'uncertain'
    
    def generate_warning(self, speculation_type: str, confidence: float) -> str:
        """Generate appropriate warning message based on speculation type and confidence"""
        warnings = {
            'hypothetical': "This is a hypothetical scenario and may not reflect reality.",
            'predictive': "This is a prediction based on current understanding and may not come to pass.",
            'opinion': "This represents a personal opinion or belief rather than established fact.",
            'possibility': "This describes a possibility rather than a certainty.",
            'uncertain': "This information has uncertain validity and should be verified."
        }
        
        base_warning = warnings.get(speculation_type, "This content requires verification.")
        
        if confidence > 0.7:
            severity = "highly speculative"
        elif confidence > 0.5:
            severity = "speculative" 
        else:
            severity = "potentially speculative"
        
        return f"⚠️ {severity.upper()}: {base_warning}"
    
    def label_dataset(self, dataset: List[Dict]) -> List[Dict]:
        """Add speculation labels to entire dataset"""
        labeled_data = []
        
        for item in dataset:
            if isinstance(item, str):
                text = item
                metadata = {}
            else:
                text = item.get('text', '')
                metadata = item
            
            speculation_analysis = self.analyze_speculation(text)
            
            labeled_item = {
                'text': text,
                'metadata': metadata,
                'speculation_analysis': speculation_analysis
            }
            
            # Add warning to text if highly speculative
            if speculation_analysis['speculation_confidence'] > 0.7:
                labeled_item['text_with_warning'] = (
                    f"{speculation_analysis['recommended_warning']}\n\n{text}"
                )
            
            labeled_data.append(labeled_item)
        
        return labeled_data
