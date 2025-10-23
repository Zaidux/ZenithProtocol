"""
Training Report Generator - Creates comprehensive training progress reports
"""

from typing import Dict, List, Any
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from enum import Enum

class ReportType(Enum):
    PROGRESS_REPORT = "progress_report"
    COMPREHENSIVE_EVALUATION = "comprehensive_evaluation"
    GAP_ANALYSIS = "gap_analysis"
    RECOMMENDATION_REPORT = "recommendation_report"

@dataclass
class TrainingMetrics:
    timestamp: datetime
    domain: str
    accuracy: float
    reasoning_score: float
    efficiency: float
    concepts_mastered: int
    training_duration: int

class TrainingReportGenerator:
    def __init__(self, output_dir: str = "training_reports"):
        self.output_dir = output_dir
        self.metrics_history = []
        
    def generate_progress_report(self, 
                               student_id: str,
                               training_period: tuple,
                               metrics: List[TrainingMetrics],
                               current_evaluation: Dict) -> Dict[str, Any]:
        """Generate progress report showing training improvements"""
        
        report = {
            'report_type': ReportType.PROGRESS_REPORT.value,
            'student_id': student_id,
            'generation_date': datetime.now().isoformat(),
            'training_period': {
                'start': training_period[0].isoformat(),
                'end': training_period[1].isoformat()
            },
            'executive_summary': self.generate_executive_summary(metrics, current_evaluation),
            'domain_performance': self.analyze_domain_performance(metrics),
            'progress_trends': self.analyze_progress_trends(metrics),
            'key_achievements': self.identify_key_achievements(metrics, current_evaluation),
            'areas_for_improvement': self.identify_improvement_areas(metrics, current_evaluation),
            'recommendations': self.generate_progress_recommendations(metrics, current_evaluation),
            'next_milestones': self.define_next_milestones(current_evaluation)
        }
        
        # Generate visualizations
        self.generate_progress_visualizations(metrics, student_id)
        
        return report
    
    def generate_comprehensive_evaluation(self,
                                        student_id: str,
                                        evaluation_results: Dict,
                                        curriculum_progress: Dict) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        
        report = {
            'report_type': ReportType.COMPREHENSIVE_EVALUATION.value,
            'student_id': student_id,
            'evaluation_date': datetime.now().isoformat(),
            'overall_assessment': self.generate_overall_assessment(evaluation_results),
            'domain_breakdown': self.breakdown_domain_performance(evaluation_results),
            'reasoning_analysis': self.analyze_reasoning_capabilities(evaluation_results),
            'knowledge_mastery': self.assess_knowledge_mastery(evaluation_results, curriculum_progress),
            'strengths_weaknesses': self.identify_strengths_weaknesses(evaluation_results),
            'comparative_analysis': self.perform_comparative_analysis(evaluation_results),
            'certification_level': self.determine_certification_level(evaluation_results)
        }
        
        return report
    
    def generate_gap_analysis(self,
                            student_id: str,
                            target_competencies: List[str],
                            current_abilities: Dict,
                            knowledge_graph) -> Dict[str, Any]:
        """Generate gap analysis between current abilities and target competencies"""
        
        gaps = self.identify_competency_gaps(target_competencies, current_abilities, knowledge_graph)
        
        report = {
            'report_type': ReportType.GAP_ANALYSIS.value,
            'student_id': student_id,
            'analysis_date': datetime.now().isoformat(),
            'target_competencies': target_competencies,
            'gap_analysis': gaps,
            'gap_prioritization': self.prioritize_gaps(gaps),
            'bridging_strategies': self.generate_bridging_strategies(gaps),
            'timeline_estimation': self.estimate_timeline(gaps),
            'resource_recommendations': self.recommend_resources(gaps, knowledge_graph)
        }
        
        return report
    
    def generate_executive_summary(self, metrics: List[TrainingMetrics], current_evaluation: Dict) -> Dict[str, Any]:
        """Generate high-level executive summary"""
        
        if not metrics:
            return {
                'overall_progress': 'No data available',
                'key_improvements': [],
                'current_status': 'Not evaluated'
            }
        
        # Calculate overall progress
        latest_metrics = metrics[-1]
        earliest_metrics = metrics[0]
        
        accuracy_improvement = latest_metrics.accuracy - earliest_metrics.accuracy
        reasoning_improvement = latest_metrics.reasoning_score - earliest_metrics.reasoning_score
        
        # Determine progress level
        if accuracy_improvement > 0.2 and reasoning_improvement > 0.2:
            progress_level = "exceptional"
        elif accuracy_improvement > 0.1 and reasoning_improvement > 0.1:
            progress_level = "good"
        elif accuracy_improvement > 0:
            progress_level = "moderate"
        else:
            progress_level = "needs_attention"
        
        return {
            'overall_progress': progress_level,
            'accuracy_improvement': round(accuracy_improvement, 3),
            'reasoning_improvement': round(reasoning_improvement, 3),
            'concepts_mastered': latest_metrics.concepts_mastered,
            'current_proficiency': self.assess_current_proficiency(current_evaluation),
            'training_efficiency': self.assess_training_efficiency(metrics)
        }
    
    def analyze_domain_performance(self, metrics: List[TrainingMetrics]) -> Dict[str, Any]:
        """Analyze performance across different domains"""
        
        domain_metrics = {}
        for metric in metrics:
            if metric.domain not in domain_metrics:
                domain_metrics[metric.domain] = []
            domain_metrics[metric.domain].append(metric)
        
        domain_analysis = {}
        for domain, domain_data in domain_metrics.items():
            accuracies = [m.accuracy for m in domain_data]
            reasoning_scores = [m.reasoning_score for m in domain_data]
            efficiencies = [m.efficiency for m in domain_data]
            
            domain_analysis[domain] = {
                'current_accuracy': accuracies[-1] if accuracies else 0,
                'accuracy_trend': self.calculate_trend(accuracies),
                'current_reasoning': reasoning_scores[-1] if reasoning_scores else 0,
                'reasoning_trend': self.calculate_trend(reasoning_scores),
                'average_efficiency': np.mean(efficiencies) if efficiencies else 0,
                'total_training_time': sum(m.training_duration for m in domain_data),
                'performance_consistency': self.assess_consistency(accuracies)
            }
        
        return domain_analysis
    
    def analyze_progress_trends(self, metrics: List[TrainingMetrics]) -> Dict[str, Any]:
        """Analyze trends in training progress"""
        
        if len(metrics) < 2:
            return {'insufficient_data': True}
        
        timestamps = [m.timestamp for m in metrics]
        accuracies = [m.accuracy for m in metrics]
        reasoning_scores = [m.reasoning_score for m in metrics]
        efficiencies = [m.efficiency for m in metrics]
        
        return {
            'accuracy_trend': self.calculate_trend(accuracies),
            'reasoning_trend': self.calculate_trend(reasoning_scores),
            'efficiency_trend': self.calculate_trend(efficiencies),
            'learning_velocity': self.calculate_learning_velocity(accuracies, timestamps),
            'plateau_detection': self.detect_learning_plateaus(accuracies),
            'consistency_analysis': self.analyze_consistency(accuracies)
        }
    
    def identify_key_achievements(self, metrics: List[TrainingMetrics], current_evaluation: Dict) -> List[str]:
        """Identify key achievements during training"""
        
        achievements = []
        
        if not metrics:
            return achievements
        
        # Check for significant accuracy improvements
        if len(metrics) >= 2:
            accuracy_gain = metrics[-1].accuracy - metrics[0].accuracy
            if accuracy_gain > 0.3:
                achievements.append(f"Major accuracy improvement: +{accuracy_gain:.1%}")
        
        # Check for concept mastery milestones
        concepts_learned = metrics[-1].concepts_mastered
        if concepts_learned >= 50:
            achievements.append(f"Mastered {concepts_learned} core concepts")
        
        # Check evaluation performance
        if current_evaluation.get('overall_score', 0) > 0.8:
            achievements.append("Achieved expert-level performance in comprehensive evaluation")
        
        # Check for consistency
        accuracies = [m.accuracy for m in metrics]
        if len(accuracies) >= 5 and np.std(accuracies[-5:]) < 0.1:
            achievements.append("Demonstrated consistent high performance")
        
        return achievements
    
    def identify_improvement_areas(self, metrics: List[TrainingMetrics], current_evaluation: Dict) -> List[Dict[str, Any]]:
        """Identify areas needing improvement"""
        
        improvement_areas = []
        
        if not metrics:
            return improvement_areas
        
        latest_metrics = metrics[-1]
        
        # Low accuracy areas
        if latest_metrics.accuracy < 0.7:
            improvement_areas.append({
                'area': 'factual_accuracy',
                'current_level': latest_metrics.accuracy,
                'target_level': 0.8,
                'priority': 'high',
                'suggested_actions': ['focused_fact_practice', 'knowledge_review']
            })
        
        # Reasoning improvement
        if latest_metrics.reasoning_score < 0.6:
            improvement_areas.append({
                'area': 'logical_reasoning',
                'current_level': latest_metrics.reasoning_score,
                'target_level': 0.75,
                'priority': 'medium',
                'suggested_actions': ['multi_step_problems', 'explanation_practice']
            })
        
        # Efficiency concerns
        if latest_metrics.efficiency < 0.5:
            improvement_areas.append({
                'area': 'response_efficiency',
                'current_level': latest_metrics.efficiency,
                'target_level': 0.7,
                'priority': 'medium',
                'suggested_actions': ['timed_practice', 'optimization_training']
            })
        
        return improvement_areas
    
    def generate_progress_recommendations(self, metrics: List[TrainingMetrics], current_evaluation: Dict) -> List[str]:
        """Generate recommendations based on progress analysis"""
        
        recommendations = []
        
        if not metrics:
            return ["Begin with foundational training in core domains"]
        
        latest_metrics = metrics[-1]
        
        # Accuracy-based recommendations
        if latest_metrics.accuracy < 0.6:
            recommendations.append("Focus on accuracy through targeted fact-checking exercises")
        elif latest_metrics.accuracy < 0.8:
            recommendations.append("Continue accuracy practice while introducing more complex scenarios")
        
        # Reasoning-based recommendations
        if latest_metrics.reasoning_score < 0.5:
            recommendations.append("Practice step-by-step reasoning with explicit justification")
        elif latest_metrics.reasoning_score < 0.7:
            recommendations.append("Work on connecting concepts across different domains")
        
        # Efficiency recommendations
        if latest_metrics.efficiency < 0.6:
            recommendations.append("Practice responding within time constraints to improve efficiency")
        
        # Progressive difficulty
        if latest_metrics.accuracy > 0.8 and latest_metrics.reasoning_score > 0.7:
            recommendations.append("Advance to expert-level problems and cross-domain challenges")
        
        return recommendations
    
    def define_next_milestones(self, current_evaluation: Dict) -> List[Dict[str, Any]]:
        """Define next achievable milestones"""
        
        current_score = current_evaluation.get('overall_score', 0)
        
        milestones = [
            {
                'milestone': 'Intermediate Proficiency',
                'target_score': 0.7,
                'current_progress': min(current_score / 0.7, 1.0),
                'estimated_time': '2-4 weeks',
                'key_activities': ['domain_foundations', 'basic_reasoning']
            },
            {
                'milestone': 'Advanced Competence',
                'target_score': 0.8,
                'current_progress': min((current_score - 0.7) / 0.1, 1.0) if current_score > 0.7 else 0.0,
                'estimated_time': '4-6 weeks',
                'key_activities': ['complex_problems', 'multi_domain_integration']
            },
            {
                'milestone': 'Expert Mastery',
                'target_score': 0.9,
                'current_progress': min((current_score - 0.8) / 0.1, 1.0) if current_score > 0.8 else 0.0,
                'estimated_time': '6-8 weeks',
                'key_activities': ['research_level_problems', 'novel_solution_generation']
            }
        ]
        
        return milestones
    
    def generate_progress_visualizations(self, metrics: List[TrainingMetrics], student_id: str):
        """Generate visualizations of training progress"""
        
        if len(metrics) < 2:
            return
        
        # Extract data for plotting
        timestamps = [m.timestamp for m in metrics]
        accuracies = [m.accuracy for m in metrics]
        reasoning_scores = [m.reasoning_score for m in metrics]
        
        # Create progress plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(timestamps, accuracies, 'b-o', label='Accuracy', linewidth=2)
        plt.plot(timestamps, reasoning_scores, 'r-s', label='Reasoning Score', linewidth=2)
        plt.title(f'Training Progress - {student_id}')
        plt.ylabel('Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        domains = list(set(m.domain for m in metrics))
        domain_accuracies = {}
        
        for domain in domains:
            domain_metrics = [m for m in metrics if m.domain == domain]
            domain_accuracies[domain] = [m.accuracy for m in domain_metrics]
        
        for domain, accs in domain_accuracies.items():
            plt.plot(timestamps[:len(accs)], accs, '--', label=domain, alpha=0.7)
        
        plt.xlabel('Time')
        plt.ylabel('Domain Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{student_id}_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Additional helper methods for comprehensive reporting
    def calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from a series of values"""
        if len(values) < 2:
            return "insufficient_data"
        
        if values[-1] > values[0] + 0.1:
            return "strong_improvement"
        elif values[-1] > values[0]:
            return "moderate_improvement"
        elif values[-1] == values[0]:
            return "stable"
        else:
            return "declining"
    
    def calculate_learning_velocity(self, scores: List[float], timestamps: List[datetime]) -> float:
        """Calculate how quickly learning is occurring"""
        if len(scores) < 2:
            return 0.0
        
        time_delta = (timestamps[-1] - timestamps[0]).total_seconds() / 3600  # hours
        score_delta = scores[-1] - scores[0]
        
        return score_delta / time_delta if time_delta > 0 else 0.0
    
    def detect_learning_plateaus(self, scores: List[float]) -> List[Dict[str, Any]]:
        """Detect periods where learning has plateaued"""
        plateaus = []
        
        if len(scores) < 5:
            return plateaus
        
        window_size = 3
        for i in range(len(scores) - window_size):
            window_scores = scores[i:i + window_size]
            if np.std(window_scores) < 0.05:  # Very little variation
                plateaus.append({
                    'start_index': i,
                    'end_index': i + window_size - 1,
                    'average_score': np.mean(window_scores)
                })
        
        return plateaus
    
    def analyze_consistency(self, scores: List[float]) -> Dict[str, Any]:
        """Analyze consistency of performance"""
        if len(scores) < 2:
            return {'consistency': 'unknown', 'volatility': 0.0}
        
        volatility = np.std(scores)
        
        if volatility < 0.1:
            consistency = 'high'
        elif volatility < 0.2:
            consistency = 'medium'
        else:
            consistency = 'low'
        
        return {
            'consistency': consistency,
            'volatility': volatility,
            'min_score': min(scores),
            'max_score': max(scores)
        }
    
    def assess_consistency(self, scores: List[float]) -> str:
        """Assess consistency of scores"""
        if len(scores) < 2:
            return "unknown"
        
        volatility = np.std(scores)
        if volatility < 0.1:
            return "very_consistent"
        elif volatility < 0.2:
            return "consistent"
        else:
            return "inconsistent"
    
    def assess_current_proficiency(self, evaluation: Dict) -> str:
        """Assess current proficiency level"""
        overall_score = evaluation.get('overall_score', 0)
        
        if overall_score > 0.9:
            return "expert"
        elif overall_score > 0.8:
            return "advanced"
        elif overall_score > 0.7:
            return "intermediate"
        elif overall_score > 0.6:
            return "beginner"
        else:
            return "novice"
    
    def assess_training_efficiency(self, metrics: List[TrainingMetrics]) -> str:
        """Assess efficiency of training process"""
        if len(metrics) < 2:
            return "unknown"
        
        total_time = sum(m.training_duration for m in metrics)
        accuracy_gain = metrics[-1].accuracy - metrics[0].accuracy
        
        if accuracy_gain <= 0:
            return "inefficient"
        
        efficiency_ratio = accuracy_gain / total_time
        
        if efficiency_ratio > 0.01:
            return "highly_efficient"
        elif efficiency_ratio > 0.005:
            return "efficient"
        else:
            return "moderately_efficient"
    
    # Placeholder methods for comprehensive evaluation
    def generate_overall_assessment(self, evaluation_results: Dict) -> Dict[str, Any]:
        return {'summary': 'Comprehensive assessment placeholder'}
    
    def breakdown_domain_performance(self, evaluation_results: Dict) -> Dict[str, Any]:
        return {'domains': 'Domain breakdown placeholder'}
    
    def analyze_reasoning_capabilities(self, evaluation_results: Dict) -> Dict[str, Any]:
        return {'reasoning': 'Reasoning analysis placeholder'}
    
    def assess_knowledge_mastery(self, evaluation_results: Dict, curriculum_progress: Dict) -> Dict[str, Any]:
        return {'mastery': 'Knowledge mastery placeholder'}
    
    def identify_strengths_weaknesses(self, evaluation_results: Dict) -> Dict[str, Any]:
        return {'strengths': [], 'weaknesses': []}
    
    def perform_comparative_analysis(self, evaluation_results: Dict) -> Dict[str, Any]:
        return {'comparison': 'Comparative analysis placeholder'}
    
    def determine_certification_level(self, evaluation_results: Dict) -> str:
        return "certified_ai_specialist"
    
    def identify_competency_gaps(self, target_competencies: List[str], current_abilities: Dict, knowledge_graph) -> List[Dict[str, Any]]:
        return [{'competency': comp, 'gap_size': 0.5} for comp in target_competencies]
    
    def prioritize_gaps(self, gaps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(gaps, key=lambda x: x.get('gap_size', 0), reverse=True)[:5]
    
    def generate_bridging_strategies(self, gaps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{'gap': gap['competency'], 'strategy': 'focused_training'} for gap in gaps[:3]]
    
    def estimate_timeline(self, gaps: List[Dict[str, Any]]) -> Dict[str, str]:
        return {'estimated_completion': '8-12 weeks', 'confidence': 'medium'}
    
    def recommend_resources(self, gaps: List[Dict[str, Any]], knowledge_graph) -> List[Dict[str, Any]]:
        return [{'gap': gap['competency'], 'resources': ['training_modules', 'practice_exercises']} for gap in gaps[:3]]
    
    def save_report(self, report: Dict[str, Any], filename: str = None):
        """Save report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{report['report_type']}_{timestamp}.json"
        
        with open(f"{self.output_dir}/{filename}", 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def load_report(self, filename: str) -> Dict[str, Any]:
        """Load report from file"""
        with open(f"{self.output_dir}/{filename}", 'r') as f:
            return json.load(f)
