"""
Progressive Evaluator - Evaluates student models from simple to complex questions
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import time

class EvaluationPhase(Enum):
    FOUNDATIONAL = "foundational"
    INTERMEDIATE = "intermediate" 
    ADVANCED = "advanced"
    EXPERT = "expert"
    HYPER_CONCEPTUAL = "hyper_conceptual"

@dataclass
class EvaluationResult:
    phase: EvaluationPhase
    question: str
    answer: str
    scores: Dict[str, float]
    feedback: str
    time_taken: float
    concepts_tested: List[str]

class ProgressiveEvaluator:
    def __init__(self, teacher_model, knowledge_graph):
        self.teacher = teacher_model
        self.kg = knowledge_graph
        self.evaluation_phases = self.define_evaluation_phases()
        
    def define_evaluation_phases(self) -> Dict[EvaluationPhase, Dict[str, Any]]:
        """Define what each evaluation phase tests"""
        return {
            EvaluationPhase.FOUNDATIONAL: {
                'description': 'Basic knowledge and recall',
                'difficulty_range': (0, 1),
                'focus': ['facts', 'definitions', 'basic_concepts'],
                'time_limit': 30,  # seconds per question
                'passing_threshold': 0.7
            },
            EvaluationPhase.INTERMEDIATE: {
                'description': 'Application and basic reasoning',
                'difficulty_range': (1, 2),
                'focus': ['application', 'simple_inference', 'problem_solving'],
                'time_limit': 60,
                'passing_threshold': 0.65
            },
            EvaluationPhase.ADVANCED: {
                'description': 'Complex reasoning and analysis',
                'difficulty_range': (2, 3),
                'focus': ['analysis', 'synthesis', 'critical_thinking'],
                'time_limit': 120,
                'passing_threshold': 0.6
            },
            EvaluationPhase.EXPERT: {
                'description': 'Expert-level synthesis and creation',
                'difficulty_range': (3, 4),
                'focus': ['synthesis', 'evaluation', 'creation'],
                'time_limit': 180,
                'passing_threshold': 0.55
            },
            EvaluationPhase.HYPER_CONCEPTUAL: {
                'description': 'Cross-domain conceptual thinking',
                'difficulty_range': (4, 5),
                'focus': ['abstraction', 'conceptual_connections', 'novel_insights'],
                'time_limit': 300,
                'passing_threshold': 0.5
            }
        }
    
    def evaluate_student(self, student_model, domains: List[str], max_phase: EvaluationPhase = None) -> Dict[str, Any]:
        """Comprehensive progressive evaluation of student model"""
        max_phase = max_phase or EvaluationPhase.HYPER_CONCEPTUAL
        phases_to_test = list(EvaluationPhase)
        phases_to_test = phases_to_test[:phases_to_test.index(max_phase) + 1]
        
        evaluation_results = {}
        overall_scores = {}
        
        for phase in phases_to_test:
            print(f"🔍 Evaluating {phase.value} phase...")
            phase_results = self.evaluate_phase(student_model, domains, phase)
            evaluation_results[phase.value] = phase_results
            
            # Calculate phase score
            phase_score = self.calculate_phase_score(phase_results)
            overall_scores[phase.value] = phase_score
            
            # Check if student passed this phase
            phase_threshold = self.evaluation_phases[phase]['passing_threshold']
            if phase_score < phase_threshold:
                print(f"❌ Student failed {phase.value} phase (score: {phase_score:.2f} < {phase_threshold})")
                break
            else:
                print(f"✅ Student passed {phase.value} phase (score: {phase_score:.2f})")
        
        # Generate comprehensive report
        final_report = self.generate_evaluation_report(evaluation_results, overall_scores)
        
        return {
            'evaluation_results': evaluation_results,
            'overall_scores': overall_scores,
            'final_report': final_report,
            'recommended_next_steps': self.get_recommended_next_steps(overall_scores)
        }
    
    def evaluate_phase(self, student_model, domains: List[str], phase: EvaluationPhase) -> List[EvaluationResult]:
        """Evaluate student in a specific phase"""
        phase_config = self.evaluation_phases[phase]
        results = []
        
        # Generate questions for this phase
        questions = self.generate_phase_questions(domains, phase, num_questions=5)
        
        for question_data in questions:
            start_time = time.time()
            
            # Get student's answer
            student_answer = student_model.answer_question(
                question_data['question'],
                domain=question_data['domain'],
                context=question_data.get('context', {})
            )
            
            time_taken = time.time() - start_time
            
            # Evaluate answer
            evaluation = self.teacher.evaluate_answer(question_data, student_answer, question_data['domain'])
            
            # Create result
            result = EvaluationResult(
                phase=phase,
                question=question_data['question'],
                answer=student_answer,
                scores=evaluation['quality_scores'],
                feedback=self.generate_feedback(evaluation, time_taken, phase_config['time_limit']),
                time_taken=time_taken,
                concepts_tested=question_data['expected_concepts']
            )
            
            results.append(result)
        
        return results
    
    def generate_phase_questions(self, domains: List[str], phase: EvaluationPhase, num_questions: int) -> List[Dict[str, Any]]:
        """Generate questions appropriate for the evaluation phase"""
        questions = []
        phase_config = self.evaluation_phases[phase]
        
        for domain in domains:
            # Generate questions at this difficulty level
            difficulty_min, difficulty_max = phase_config['difficulty_range']
            target_difficulty = (difficulty_min + difficulty_max) / 2
            
            for _ in range(num_questions // len(domains)):
                question = self.teacher.generate_question(
                    domain=domain,
                    difficulty=int(target_difficulty),
                    context={'phase': phase.value, 'focus_areas': phase_config['focus']}
                )
                questions.append(question)
        
        return questions
    
    def calculate_phase_score(self, phase_results: List[EvaluationResult]) -> float:
        """Calculate overall score for a phase"""
        if not phase_results:
            return 0.0
        
        # Weight different aspects
        accuracy_scores = [result.scores['accuracy'] for result in phase_results]
        depth_scores = [result.scores['depth'] for result in phase_results]
        time_scores = [self.calculate_time_score(result.time_taken, result.phase) for result in phase_results]
        
        # Overall phase score
        accuracy_avg = np.mean(accuracy_scores)
        depth_avg = np.mean(depth_scores)
        time_avg = np.mean(time_scores)
        
        return (accuracy_avg * 0.5 + depth_avg * 0.3 + time_avg * 0.2)
    
    def calculate_time_score(self, time_taken: float, phase: EvaluationPhase) -> float:
        """Calculate score based on time efficiency"""
        time_limit = self.evaluation_phases[phase]['time_limit']
        if time_taken <= time_limit:
            return 1.0
        else:
            # Exponential decay for overtime
            overtime = time_taken - time_limit
            penalty = min(overtime / time_limit, 1.0)  # Max penalty of 50%
            return 1.0 - (penalty * 0.5)
    
    def generate_feedback(self, evaluation: Dict, time_taken: float, time_limit: float) -> str:
        """Generate constructive feedback for the student"""
        feedback_parts = []
        
        # Accuracy feedback
        accuracy = evaluation['quality_scores']['accuracy']
        if accuracy > 0.8:
            feedback_parts.append("Excellent accuracy in your response.")
        elif accuracy > 0.6:
            feedback_parts.append("Good accuracy, but there's room for improvement.")
        else:
            feedback_parts.append("Focus on improving the factual correctness of your answers.")
        
        # Depth feedback
        depth = evaluation['quality_scores']['depth']
        if depth > 0.8:
            feedback_parts.append("Your answer shows impressive depth of understanding.")
        elif depth > 0.6:
            feedback_parts.append("Consider adding more detailed analysis to your answers.")
        else:
            feedback_parts.append("Try to explore concepts more deeply in your responses.")
        
        # Time feedback
        if time_taken > time_limit:
            feedback_parts.append(f"You took {time_taken:.1f}s (limit: {time_limit}s). Work on responding more efficiently.")
        else:
            feedback_parts.append(f"Good time management ({time_taken:.1f}s).")
        
        # Concept mastery feedback
        concept_mastery = evaluation['concept_mastery']
        if concept_mastery < 0.5:
            feedback_parts.append("Review the core concepts tested in this question.")
        
        # Strengths and weaknesses
        if evaluation['strengths']:
            feedback_parts.append(f"Strengths: {', '.join(evaluation['strengths'][:2])}")
        if evaluation['identified_weaknesses']:
            feedback_parts.append(f"Areas for improvement: {', '.join(evaluation['identified_weaknesses'][:2])}")
        
        return " ".join(feedback_parts)
    
    def generate_evaluation_report(self, evaluation_results: Dict, overall_scores: Dict) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        report = {
            'summary': {},
            'phase_breakdown': {},
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        # Calculate overall statistics
        all_scores = []
        all_times = []
        concepts_mastered = set()
        concepts_missed = set()
        
        for phase_name, phase_results in evaluation_results.items():
            phase_scores = [result.scores['accuracy'] for result in phase_results]
            phase_times = [result.time_taken for result in phase_results]
            
            report['phase_breakdown'][phase_name] = {
                'average_score': np.mean(phase_scores) if phase_scores else 0,
                'average_time': np.mean(phase_times) if phase_times else 0,
                'questions_attempted': len(phase_results),
                'phase_score': overall_scores.get(phase_name, 0)
            }
            
            all_scores.extend(phase_scores)
            all_times.extend(phase_times)
            
            # Track concepts
            for result in phase_results:
                concepts_tested = set(result.concepts_tested)
                if result.scores['accuracy'] > 0.7:
                    concepts_mastered.update(concepts_tested)
                else:
                    concepts_missed.update(concepts_tested)
        
        report['summary'] = {
            'overall_score': np.mean(all_scores) if all_scores else 0,
            'average_response_time': np.mean(all_times) if all_times else 0,
            'total_questions': len(all_scores),
            'concepts_mastered': len(concepts_mastered),
            'concepts_needing_work': len(concepts_missed),
            'efficiency_rating': self.calculate_efficiency_rating(all_times)
        }
        
        # Identify strengths and weaknesses
        report['strengths'] = self.identify_strengths(evaluation_results)
        report['weaknesses'] = self.identify_weaknesses(evaluation_results)
        report['recommendations'] = self.generate_recommendations(report)
        
        return report
    
    def calculate_efficiency_rating(self, times: List[float]) -> str:
        """Calculate efficiency rating based on response times"""
        avg_time = np.mean(times) if times else 0
        if avg_time < 30:
            return "excellent"
        elif avg_time < 60:
            return "good"
        elif avg_time < 120:
            return "average"
        else:
            return "needs_improvement"
    
    def identify_strengths(self, evaluation_results: Dict) -> List[str]:
        """Identify student's key strengths"""
        strengths = []
        
        high_scoring_phases = []
        for phase_name, phase_results in evaluation_results.items():
            phase_scores = [result.scores['accuracy'] for result in phase_results]
            if phase_scores and np.mean(phase_scores) > 0.8:
                high_scoring_phases.append(phase_name)
        
        if high_scoring_phases:
            strengths.append(f"Excels in {', '.join(high_scoring_phases)} phases")
        
        # Check for consistent performance
        all_scores = []
        for phase_results in evaluation_results.values():
            all_scores.extend([result.scores['accuracy'] for result in phase_results])
        
        if all_scores and np.std(all_scores) < 0.2:
            strengths.append("Consistent performance across different question types")
        
        return strengths
    
    def identify_weaknesses(self, evaluation_results: Dict) -> List[str]:
        """Identify areas needing improvement"""
        weaknesses = []
        
        low_scoring_phases = []
        for phase_name, phase_results in evaluation_results.items():
            phase_scores = [result.scores['accuracy'] for result in phase_results]
            if phase_scores and np.mean(phase_scores) < 0.6:
                low_scoring_phases.append(phase_name)
        
        if low_scoring_phases:
            weaknesses.append(f"Struggles with {', '.join(low_scoring_phases)} level questions")
        
        return weaknesses
    
    def generate_recommendations(self, report: Dict) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        summary = report['summary']
        
        if summary['overall_score'] < 0.7:
            recommendations.append("Focus on foundational knowledge before advancing to complex topics")
        
        if summary['efficiency_rating'] in ['average', 'needs_improvement']:
            recommendations.append("Practice responding more efficiently to time-constrained questions")
        
        if summary['concepts_needing_work'] > summary['concepts_mastered']:
            recommendations.append("Review core concepts where gaps in understanding were identified")
        
        if report['weaknesses']:
            for weakness in report['weaknesses']:
                recommendations.append(f"Targeted practice for: {weakness}")
        
        return recommendations
    
    def get_recommended_next_steps(self, overall_scores: Dict) -> List[str]:
        """Get recommended next steps based on evaluation results"""
        next_steps = []
        
        # Find highest phase passed
        phases = list(EvaluationPhase)
        highest_passed = EvaluationPhase.FOUNDATIONAL
        
        for phase in phases:
            phase_score = overall_scores.get(phase.value, 0)
            phase_threshold = self.evaluation_phases[phase]['passing_threshold']
            if phase_score >= phase_threshold:
                highest_passed = phase
        
        # Recommend next phase
        next_phase_index = phases.index(highest_passed) + 1
        if next_phase_index < len(phases):
            next_phase = phases[next_phase_index]
            next_steps.append(f"Advance to {next_phase.value} level training")
        else:
            next_steps.append("All evaluation phases completed! Consider specialized advanced training")
        
        return next_steps
