"""
Multi-Step Assessor - Evaluates complex reasoning through step-by-step analysis
"""

from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re

class ReasoningStepType(Enum):
    PREMISE_IDENTIFICATION = "premise_identification"
    INFERENCE_STEP = "inference_step"
    CONCLUSION_DRAWING = "conclusion_drawing"
    ASSUMPTION_CHECKING = "assumption_checking"
    COUNTERFACTUAL_THINKING = "counterfactual_thinking"

@dataclass
class ReasoningStep:
    step_type: ReasoningStepType
    content: str
    correctness: float
    relevance: float
    clarity: float

class MultiStepAssessor:
    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph
        self.reasoning_indicators = self.setup_reasoning_indicators()
        
    def setup_reasoning_indicators(self) -> Dict[ReasoningStepType, List[str]]:
        """Setup linguistic indicators for different reasoning steps"""
        return {
            ReasoningStepType.PREMISE_IDENTIFICATION: [
                'given that', 'since', 'because', 'as', 'based on', 'from the fact that'
            ],
            ReasoningStepType.INFERENCE_STEP: [
                'therefore', 'thus', 'hence', 'so', 'consequently', 'it follows that'
            ],
            ReasoningStepType.CONCLUSION_DRAWING: [
                'in conclusion', 'to summarize', 'overall', 'the main point is',
                'this shows that', 'this demonstrates that'
            ],
            ReasoningStepType.ASSUMPTION_CHECKING: [
                'assuming that', 'if we assume', 'presuming', 'supposing that'
            ],
            ReasoningStepType.COUNTERFACTUAL_THINKING: [
                'if instead', 'alternatively', 'on the other hand', 'conversely'
            ]
        }
    
    def assess_reasoning_chain(self, question: str, answer: str, domain: str) -> Dict[str, Any]:
        """Comprehensive assessment of multi-step reasoning"""
        
        # Parse reasoning steps
        reasoning_steps = self.parse_reasoning_steps(answer)
        
        # Evaluate each step
        evaluated_steps = []
        for step in reasoning_steps:
            evaluated_step = self.evaluate_reasoning_step(step, question, domain)
            evaluated_steps.append(evaluated_step)
        
        # Assess overall reasoning quality
        overall_quality = self.assess_overall_reasoning_quality(evaluated_steps, question, domain)
        
        # Identify reasoning gaps
        reasoning_gaps = self.identify_reasoning_gaps(evaluated_steps, question, domain)
        
        # Generate improvement suggestions
        improvements = self.generate_improvement_suggestions(evaluated_steps, reasoning_gaps)
        
        return {
            'reasoning_steps': evaluated_steps,
            'overall_quality': overall_quality,
            'reasoning_gaps': reasoning_gaps,
            'improvement_suggestions': improvements,
            'reasoning_score': self.calculate_reasoning_score(evaluated_steps, overall_quality),
            'step_coherence': self.assess_step_coherence(evaluated_steps),
            'logical_flow': self.assess_logical_flow(evaluated_steps)
        }
    
    def parse_reasoning_steps(self, answer: str) -> List[ReasoningStep]:
        """Parse answer into distinct reasoning steps"""
        steps = []
        
        # Split by common step indicators
        step_delimiters = ['.', ';', '\n', '•', '- ']
        
        # First, try to split by explicit step markers
        step_patterns = [
            r'Step\s+\d+[:.]?\s*',
            r'\d+\.\s*',
            r'•\s*',
            r'-\s*',
            r'First[,.]?\s*|Second[,.]?\s*|Third[,.]?\s*|Finally[,.]?\s*'
        ]
        
        current_text = answer
        step_number = 1
        
        while current_text.strip():
            # Look for the next step marker
            earliest_match = None
            earliest_pos = len(current_text)
            
            for pattern in step_patterns:
                match = re.search(pattern, current_text, re.IGNORECASE)
                if match and match.start() < earliest_pos:
                    earliest_match = match
                    earliest_pos = match.start()
            
            if earliest_match:
                # Extract step before marker
                if earliest_pos > 0:
                    step_content = current_text[:earliest_pos].strip()
                    if step_content:
                        step_type = self.classify_step_type(step_content)
                        steps.append(ReasoningStep(
                            step_type=step_type,
                            content=step_content,
                            correctness=0.0,  # To be evaluated
                            relevance=0.0,
                            clarity=0.0
                        ))
                
                # Move past this marker
                current_text = current_text[earliest_match.end():]
                step_number += 1
            else:
                # No more markers, take the rest as final step
                if current_text.strip():
                    step_type = self.classify_step_type(current_text.strip())
                    steps.append(ReasoningStep(
                        step_type=step_type,
                        content=current_text.strip(),
                        correctness=0.0,
                        relevance=0.0,
                        clarity=0.0
                    ))
                break
        
        # If no steps were found with markers, split by sentences
        if not steps:
            sentences = re.split(r'[.!?]+', answer)
            for sentence in sentences:
                if sentence.strip():
                    step_type = self.classify_step_type(sentence.strip())
                    steps.append(ReasoningStep(
                        step_type=step_type,
                        content=sentence.strip(),
                        correctness=0.0,
                        relevance=0.0,
                        clarity=0.0
                    ))
        
        return steps
    
    def classify_step_type(self, step_content: str) -> ReasoningStepType:
        """Classify the type of reasoning step"""
        step_lower = step_content.lower()
        
        # Check for premise indicators
        if any(indicator in step_lower for indicator in 
               self.reasoning_indicators[ReasoningStepType.PREMISE_IDENTIFICATION]):
            return ReasoningStepType.PREMISE_IDENTIFICATION
        
        # Check for inference indicators
        if any(indicator in step_lower for indicator in 
               self.reasoning_indicators[ReasoningStepType.INFERENCE_STEP]):
            return ReasoningStepType.INFERENCE_STEP
        
        # Check for conclusion indicators
        if any(indicator in step_lower for indicator in 
               self.reasoning_indicators[ReasoningStepType.CONCLUSION_DRAWING]):
            return ReasoningStepType.CONCLUSION_DRAWING
        
        # Check for assumption indicators
        if any(indicator in step_lower for indicator in 
               self.reasoning_indicators[ReasoningStepType.ASSUMPTION_CHECKING]):
            return ReasoningStepType.ASSUMPTION_CHECKING
        
        # Check for counterfactual indicators
        if any(indicator in step_lower for indicator in 
               self.reasoning_indicators[ReasoningStepType.COUNTERFACTUAL_THINKING]):
            return ReasoningStepType.COUNTERFACTUAL_THINKING
        
        # Default to inference step
        return ReasoningStepType.INFERENCE_STEP
    
    def evaluate_reasoning_step(self, step: ReasoningStep, question: str, domain: str) -> ReasoningStep:
        """Evaluate a single reasoning step"""
        # Evaluate correctness (domain-specific)
        step.correctness = self.evaluate_step_correctness(step, domain)
        
        # Evaluate relevance to the question
        step.relevance = self.evaluate_step_relevance(step, question)
        
        # Evaluate clarity of expression
        step.clarity = self.evaluate_step_clarity(step)
        
        return step
    
    def evaluate_step_correctness(self, step: ReasoningStep, domain: str) -> float:
        """Evaluate factual and logical correctness of a step"""
        # This would integrate with your knowledge graph
        # For now, use simple heuristic-based evaluation
        
        correctness_indicators = {
            'clear_facts': 0.3,
            'logical_connectors': 0.2,
            'domain_appropriate': 0.3,
            'no_contradictions': 0.2
        }
        
        score = 0.0
        step_lower = step.content.lower()
        
        # Check for clear factual statements
        if any(word in step_lower for word in ['is', 'are', 'has', 'have', 'contains']):
            score += correctness_indicators['clear_facts']
        
        # Check for logical connectors
        if any(connector in step_lower for connector in ['because', 'therefore', 'thus', 'since']):
            score += correctness_indicators['logical_connectors']
        
        # Check for domain-appropriate terminology
        domain_terms = self.get_domain_terms(domain)
        if any(term in step_lower for term in domain_terms):
            score += correctness_indicators['domain_appropriate']
        
        # No obvious contradictions (simplified check)
        contradiction_terms = ['but not', 'however', 'although', 'despite']
        if not any(term in step_lower for term in contradiction_terms):
            score += correctness_indicators['no_contradictions']
        
        return min(score, 1.0)
    
    def evaluate_step_relevance(self, step: ReasoningStep, question: str) -> float:
        """Evaluate how relevant the step is to the original question"""
        question_terms = set(question.lower().split())
        step_terms = set(step.content.lower().split())
        
        # Calculate term overlap
        overlap = len(question_terms.intersection(step_terms))
        total_unique = len(question_terms.union(step_terms))
        
        if total_unique > 0:
            base_relevance = overlap / total_unique
        else:
            base_relevance = 0.0
        
        # Boost for reasoning steps that connect concepts
        reasoning_boost = 0.2 if step.step_type in [
            ReasoningStepType.INFERENCE_STEP, 
            ReasoningStepType.CONCLUSION_DRAWING
        ] else 0.0
        
        return min(base_relevance + reasoning_boost, 1.0)
    
    def evaluate_step_clarity(self, step: ReasoningStep) -> float:
        """Evaluate clarity and comprehensibility of the step"""
        clarity_penalties = 0.0
        max_penalties = 5.0
        
        # Penalize very long sentences
        if len(step.content.split()) > 30:
            clarity_penalties += 1.0
        
        # Penalize complex sentence structures
        if step.content.count(',') > 3:
            clarity_penalties += 0.5
        
        # Penalize vague language
        vague_terms = ['thing', 'stuff', 'something', 'somehow', 'kind of']
        if any(term in step.content.lower() for term in vague_terms):
            clarity_penalties += 1.0
        
        # Penalize passive voice (simplified check)
        passive_indicators = ['is done', 'was made', 'are given']
        if any(indicator in step.content.lower() for indicator in passive_indicators):
            clarity_penalties += 0.5
        
        clarity_score = 1.0 - (clarity_penalties / max_penalties)
        return max(clarity_score, 0.0)
    
    def assess_overall_reasoning_quality(self, steps: List[ReasoningStep], question: str, domain: str) -> Dict[str, float]:
        """Assess the overall quality of the reasoning chain"""
        if not steps:
            return {
                'completeness': 0.0,
                'soundness': 0.0,
                'depth': 0.0,
                'originality': 0.0
            }
        
        # Completeness: Are all necessary steps present?
        completeness = self.assess_reasoning_completeness(steps, question, domain)
        
        # Soundness: How logically sound is the reasoning?
        soundness = self.assess_reasoning_soundness(steps)
        
        # Depth: How deep does the reasoning go?
        depth = self.assess_reasoning_depth(steps)
        
        # Originality: How creative or novel is the approach?
        originality = self.assess_reasoning_originality(steps, domain)
        
        return {
            'completeness': completeness,
            'soundness': soundness,
            'depth': depth,
            'originality': originality
        }
    
    def assess_reasoning_completeness(self, steps: List[ReasoningStep], question: str, domain: str) -> float:
        """Assess if reasoning covers all necessary aspects"""
        required_step_types = self.get_required_step_types(question, domain)
        
        present_types = set(step.step_type for step in steps)
        missing_types = required_step_types - present_types
        
        completeness = 1.0 - (len(missing_types) / max(len(required_step_types), 1))
        return completeness
    
    def get_required_step_types(self, question: str, domain: str) -> set:
        """Determine what types of reasoning steps are required"""
        base_types = {
            ReasoningStepType.PREMISE_IDENTIFICATION,
            ReasoningStepType.INFERENCE_STEP,
            ReasoningStepType.CONCLUSION_DRAWING
        }
        
        question_lower = question.lower()
        
        # Add assumption checking for hypothetical questions
        if any(word in question_lower for word in ['if', 'suppose', 'assume']):
            base_types.add(ReasoningStepType.ASSUMPTION_CHECKING)
        
        # Add counterfactual thinking for comparative questions
        if any(word in question_lower for word in ['instead', 'alternative', 'compare']):
            base_types.add(ReasoningStepType.COUNTERFACTUAL_THINKING)
        
        return base_types
    
    def assess_reasoning_soundness(self, steps: List[ReasoningStep]) -> float:
        """Assess logical soundness of the reasoning chain"""
        if len(steps) < 2:
            return 0.3  # Minimal soundness for single-step reasoning
        
        # Check for logical progression
        soundness_scores = []
        for i in range(len(steps) - 1):
            current_step = steps[i]
            next_step = steps[i + 1]
            
            # Check if steps connect logically
            connection_strength = self.assess_step_connection(current_step, next_step)
            soundness_scores.append(connection_strength)
        
        return sum(soundness_scores) / len(soundness_scores) if soundness_scores else 0.5
    
    def assess_step_connection(self, step1: ReasoningStep, step2: ReasoningStep) -> float:
        """Assess how well two consecutive steps connect"""
        # Check for conceptual overlap
        step1_terms = set(step1.content.lower().split())
        step2_terms = set(step2.content.lower().split())
        term_overlap = len(step1_terms.intersection(step2_terms))
        
        # Check for logical connectors between steps
        logical_connectors = ['therefore', 'thus', 'hence', 'so', 'because', 'since']
        step2_has_connector = any(connector in step2.content.lower() for connector in logical_connectors)
        
        base_score = min(term_overlap / 5.0, 0.5)  # Normalize term overlap
        if step2_has_connector:
            base_score += 0.3
        
        return min(base_score, 1.0)
    
    def assess_reasoning_depth(self, steps: List[ReasoningStep]) -> float:
        """Assess depth of reasoning (how many layers of why/how)"""
        depth_indicators = 0
        max_indicators = 5
        
        for step in steps:
            # Questions in reasoning indicate depth
            if '?' in step.content:
                depth_indicators += 1
            
            # Explanations of mechanisms indicate depth
            if any(word in step.content.lower() for word in ['because', 'since', 'due to']):
                depth_indicators += 1
            
            # Multiple perspectives indicate depth
            if any(word in step.content.lower() for word in ['however', 'alternatively', 'on the other hand']):
                depth_indicators += 1
        
        return min(depth_indicators / max_indicators, 1.0)
    
    def assess_reasoning_originality(self, steps: List[ReasoningStep], domain: str) -> float:
        """Assess originality and creativity of reasoning"""
        # Simplified originality assessment
        originality_indicators = 0
        max_indicators = 4
        
        for step in steps:
            # Novel connections between concepts
            if self.contains_novel_connection(step, domain):
                originality_indicators += 1
            
            # Unexpected analogies or metaphors
            if any(word in step.content.lower() for word in ['like', 'similar to', 'analogous']):
                originality_indicators += 1
        
        return min(originality_indicators / max_indicators, 1.0)
    
    def contains_novel_connection(self, step: ReasoningStep, domain: str) -> bool:
        """Check if step contains novel conceptual connections"""
        # This would integrate with your knowledge graph to find novel connections
        # For now, use simple heuristic
        cross_domain_terms = self.get_cross_domain_terms(domain)
        return any(term in step.content.lower() for term in cross_domain_terms)
    
    def identify_reasoning_gaps(self, steps: List[ReasoningStep], question: str, domain: str) -> List[str]:
        """Identify specific gaps in the reasoning process"""
        gaps = []
        
        # Check for missing step types
        required_types = self.get_required_step_types(question, domain)
        present_types = set(step.step_type for step in steps)
        missing_types = required_types - present_types
        
        for missing_type in missing_types:
            gaps.append(f"Missing {missing_type.value.replace('_', ' ')}")
        
        # Check for logical jumps between steps
        for i in range(len(steps) - 1):
            connection_strength = self.assess_step_connection(steps[i], steps[i + 1])
            if connection_strength < 0.3:
                gaps.append(f"Weak logical connection between steps {i+1} and {i+2}")
        
        # Check for unanswered aspects of the question
        question_aspects = self.identify_question_aspects(question)
        covered_aspects = self.identify_covered_aspects(steps)
        unanswered_aspects = question_aspects - covered_aspects
        
        for aspect in unanswered_aspects:
            gaps.append(f"Unaddressed aspect: {aspect}")
        
        return gaps
    
    def assess_step_coherence(self, steps: List[ReasoningStep]) -> float:
        """Assess how well steps fit together coherently"""
        if len(steps) <= 1:
            return 1.0  # Single step is always coherent
        
        coherence_scores = []
        for i in range(len(steps) - 1):
            coherence = self.assess_step_connection(steps[i], steps[i + 1])
            coherence_scores.append(coherence)
        
        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.5
    
    def assess_logical_flow(self, steps: List[ReasoningStep]) -> str:
        """Assess the overall logical flow quality"""
        avg_correctness = sum(step.correctness for step in steps) / len(steps) if steps else 0
        coherence = self.assess_step_coherence(steps)
        
        overall_quality = (avg_correctness + coherence) / 2
        
        if overall_quality > 0.8:
            return "excellent"
        elif overall_quality > 0.6:
            return "good"
        elif overall_quality > 0.4:
            return "adequate"
        else:
            return "needs_improvement"
    
    def calculate_reasoning_score(self, steps: List[ReasoningStep], overall_quality: Dict) -> float:
        """Calculate overall reasoning score"""
        if not steps:
            return 0.0
        
        # Component weights
        step_scores = sum(
            (step.correctness * 0.4 + step.relevance * 0.3 + step.clarity * 0.3) 
            for step in steps
        ) / len(steps)
        
        overall_scores = (
            overall_quality['completeness'] * 0.3 +
            overall_quality['soundness'] * 0.4 +
            overall_quality['depth'] * 0.2 +
            overall_quality['originality'] * 0.1
        )
        
        return (step_scores * 0.6 + overall_scores * 0.4)
    
    def generate_improvement_suggestions(self, steps: List[ReasoningStep], gaps: List[str]) -> List[str]:
        """Generate specific suggestions for improving reasoning"""
        suggestions = []
        
        # Suggestions based on gaps
        for gap in gaps:
            if "Missing premise" in gap:
                suggestions.append("Start by clearly stating the known facts or assumptions")
            elif "Missing inference" in gap:
                suggestions.append("Explain the logical connections between your points more explicitly")
            elif "Missing conclusion" in gap:
                suggestions.append("Always end with a clear conclusion that directly answers the question")
            elif "Weak logical connection" in gap:
                suggestions.append("Use logical connectors like 'therefore', 'because', or 'thus' to show relationships")
        
        # Suggestions based on step quality
        low_clarity_steps = [step for step in steps if step.clarity < 0.5]
        if low_clarity_steps:
            suggestions.append("Work on expressing ideas more clearly and concisely")
        
        low_relevance_steps = [step for step in steps if step.relevance < 0.4]
        if low_relevance_steps:
            suggestions.append("Ensure every step directly contributes to answering the question")
        
        # General reasoning suggestions
        if len(steps) < 3:
            suggestions.append("Break down complex problems into more detailed steps")
        
        if not any(step.step_type == ReasoningStepType.ASSUMPTION_CHECKING for step in steps):
            suggestions.append("Consider explicitly stating and checking your assumptions")
        
        return suggestions[:5]  # Return top 5 suggestions
    
    # Helper methods
    def get_domain_terms(self, domain: str) -> List[str]:
        """Get domain-specific terminology"""
        term_map = {
            'mathematics': ['calculate', 'equation', 'formula', 'solve', 'proof'],
            'coding': ['function', 'algorithm', 'variable', 'loop', 'condition'],
            'science': ['hypothesis', 'experiment', 'theory', 'evidence', 'conclusion']
        }
        return term_map.get(domain, [])
    
    def get_cross_domain_terms(self, domain: str) -> List[str]:
        """Get terms from other domains that might indicate novel connections"""
        cross_domain_map = {
            'mathematics': ['art', 'music', 'nature', 'beauty', 'symmetry'],
            'coding': ['poetry', 'architecture', 'language', 'creativity'],
            'science': ['philosophy', 'ethics', 'society', 'future']
        }
        return cross_domain_map.get(domain, [])
    
    def identify_question_aspects(self, question: str) -> set:
        """Identify key aspects that need to be addressed in the answer"""
        aspects = set()
        question_lower = question.lower()
        
        if 'how' in question_lower:
            aspects.add('mechanism')
        if 'why' in question_lower:
            aspects.add('causality')
        if 'compare' in question_lower:
            aspects.add('comparison')
        if 'example' in question_lower:
            aspects.add('illustration')
        
        return aspects
    
    def identify_covered_aspects(self, steps: List[ReasoningStep]) -> set:
        """Identify which question aspects are covered in the reasoning"""
        aspects = set()
        
        for step in steps:
            step_lower = step.content.lower()
            
            if any(word in step_lower for word in ['by', 'through', 'method']):
                aspects.add('mechanism')
            if any(word in step_lower for word in ['because', 'since', 'reason']):
                aspects.add('causality')
            if any(word in step_lower for word in ['compared', 'similar', 'different']):
                aspects.add('comparison')
            if any(word in step_lower for word in ['for example', 'such as', 'instance']):
                aspects.add('illustration')
        
        return aspects
