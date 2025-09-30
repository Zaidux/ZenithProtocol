# /src/modules/self_evolving_sparse_attention.py

"""
Self-Evolving Sparse Attention - The pinnacle of Zenith's attention optimization.
Combines CKG guidance with evolutionary pattern discovery and HCT-driven innovation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from datetime import datetime, timedelta

from .ckg_guided_sparse_attention import CKGSparseAttention
from .self_evolving_sparsity import PatternEvolutionEngine, AttentionPatternGene
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph
from ..models.hyper_conceptual_thinking import ConceptDiscoveryEngine

class SelfEvolvingSparseAttention(CKGSparseAttention):
    """
    Self-evolving sparse attention that discovers and optimizes its own patterns.
    Integrates CKG guidance, evolutionary algorithms, and hyper-conceptual thinking.
    """
    
    def __init__(self, 
                 dim: int,
                 num_heads: int = 8,
                 ckg: ConceptualKnowledgeGraph = None,
                 sparsity_ratio: float = 0.15,
                 max_seq_length: int = 4096,
                 evolution_interval: int = 100,  # Evolve every 100 steps
                 exploration_rate: float = 0.1):
        
        super().__init__(dim, num_heads, ckg, sparsity_ratio, max_seq_length)
        
        self.evolution_interval = evolution_interval
        self.exploration_rate = exploration_rate
        self.optimization_steps = 0
        
        # Evolutionary engine
        self.evolution_engine = PatternEvolutionEngine(
            population_size=30,
            elite_count=3,
            mutation_rate=0.2,
            crossover_rate=0.4
        )
        
        # HCT integration for pattern innovation
        self.hct_engine = ConceptDiscoveryEngine(ckg, self)
        
        # Performance tracking for evolution
        self.pattern_performance_log = {}
        self.context_pattern_map = {}  # Map contexts to best patterns
        self.performance_feedback_buffer = deque(maxlen=1000)
        
        # Adaptive learning parameters
        self.adaptive_params = {
            'current_exploration_rate': exploration_rate,
            'evolution_urgency': 0.0,
            'pattern_stagnation_count': 0,
            'last_major_improvement': datetime.now()
        }
    
    def forward(self, 
                x: torch.Tensor,
                context: Dict = None,
                return_attention_weights: bool = False,
                enable_evolution: bool = True) -> torch.Tensor:
        """
        Forward pass with self-evolving sparse attention.
        """
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Get or evolve attention pattern
        attention_mask, pattern_gene, pattern_info = self._get_evolved_pattern(
            x, seq_len, context, enable_evolution
        )
        
        # Compute sparse attention
        attn_output, attn_weights = self._sparse_attention(
            q, k, v, attention_mask, seq_len
        )
        
        # Collect performance feedback for evolution
        if enable_evolution:
            self._collect_performance_feedback(
                pattern_gene, attn_weights, x, context, pattern_info
            )
        
        # Evolve if interval reached
        self.optimization_steps += 1
        if (enable_evolution and 
            self.optimization_steps % self.evolution_interval == 0):
            self._trigger_evolution(context, seq_len)
        
        # Combine heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.dim
        )
        output = self.out_proj(attn_output)
        
        # Update statistics
        self._update_attention_stats(seq_len, attention_mask)
        
        if return_attention_weights:
            return output, attn_weights, {
                'pattern_gene': pattern_gene,
                'pattern_info': pattern_info,
                'evolution_step': self.optimization_steps
            }
        return output
    
    def _get_evolved_pattern(self,
                           x: torch.Tensor,
                           seq_len: int,
                           context: Dict,
                           enable_evolution: bool) -> Tuple[torch.Tensor, AttentionPatternGene, Dict]:
        """
        Get evolved attention pattern, potentially exploring new patterns.
        """
        context_key = self._generate_context_key(context, seq_len)
        pattern_info = {
            'context_key': context_key,
            'exploration_used': False,
            'pattern_source': 'evolution'
        }
        
        # Check if we have a cached pattern for this context
        if context_key in self.context_pattern_map and not enable_evolution:
            best_gene = self.context_pattern_map[context_key]
            pattern_mask = self._expand_pattern_to_mask(best_gene.pattern_matrix, seq_len)
            return pattern_mask, best_gene, pattern_info
        
        # Get best pattern from evolution engine
        best_gene = self.evolution_engine.get_best_pattern(seq_len, context)
        
        # Exploration: occasionally try new patterns
        if (enable_evolution and 
            random.random() < self.adaptive_params['current_exploration_rate']):
            exploration_gene = self._explore_new_pattern(seq_len, context)
            if exploration_gene is not None:
                best_gene = exploration_gene
                pattern_info['exploration_used'] = True
                pattern_info['pattern_source'] = 'exploration'
        
        # Expand pattern to full attention mask
        pattern_mask = self._expand_pattern_to_mask(best_gene.pattern_matrix, seq_len)
        
        # Cache the pattern
        if context_key not in self.context_pattern_map:
            self.context_pattern_map[context_key] = best_gene
        
        return pattern_mask, best_gene, pattern_info
    
    def _explore_new_pattern(self, seq_len: int, context: Dict) -> Optional[AttentionPatternGene]:
        """Explore new pattern possibilities using HCT."""
        try:
            # Use HCT to discover novel patterns
            exploration_context = {**(context or {}), 'exploration': True}
            
            # Generate conceptual features for pattern discovery
            conceptual_features = self._extract_conceptual_features(
                torch.randn(1, seq_len, self.dim),  # Dummy input for feature extraction
                exploration_context
            )
            
            # Use HCT to discover novel patterns
            fused_repr = torch.randn(1, self.dim)  # Simplified
            bonus, concept_name = self.hct_engine.analyze_for_new_concepts(
                fused_repr, 0.5, 0.0, context.get('domain', 'general')
            )
            
            if bonus > 2.0:  # Significant discovery
                # Create novel pattern based on HCT discovery
                novel_pattern = self._create_hct_inspired_pattern(
                    seq_len, context, concept_name
                )
                
                novel_gene = AttentionPatternGene(
                    novel_pattern,
                    f"hct_{concept_name}",
                    {**(context or {}), 'hct_discovery': concept_name},
                    bonus * 0.1  # Initial fitness based on HCT bonus
                )
                
                print(f"[HCT Pattern Discovery] New pattern inspired by '{concept_name}' "
                      f"with bonus {bonus:.2f}")
                
                return novel_gene
        
        except Exception as e:
            print(f"HCT pattern exploration failed: {e}")
        
        return None
    
    def _create_hct_inspired_pattern(self, 
                                   seq_len: int, 
                                   context: Dict, 
                                   concept_name: str) -> torch.Tensor:
        """Create novel pattern inspired by HCT discovery."""
        pattern = torch.zeros(seq_len, seq_len)
        
        # Different pattern strategies based on concept type
        if 'symmetry' in concept_name.lower():
            # Symmetrical pattern
            for i in range(seq_len):
                # Mirror attention
                mirror_idx = seq_len - 1 - i
                pattern[i, max(0, i-8):min(seq_len, i+9)] = 1
                if mirror_idx != i:
                    pattern[i, max(0, mirror_idx-4):min(seq_len, mirror_idx+5)] = 1
        
        elif 'hierarchy' in concept_name.lower():
            # Hierarchical pattern
            for i in range(seq_len):
                # Multiple levels of attention
                pattern[i, max(0, i-4):min(seq_len, i+5)] = 1  # Local
                if i % 4 == 0:
                    pattern[i, ::4] = 1  # Level 1
                if i % 16 == 0:
                    pattern[i, ::16] = 1  # Level 2
        
        elif 'cluster' in concept_name.lower():
            # Clustered pattern
            cluster_size = min(16, seq_len // 4)
            for i in range(0, seq_len, cluster_size):
                end_i = min(i + cluster_size, seq_len)
                pattern[i:end_i, i:end_i] = 1
                # Connect clusters
                if i + cluster_size < seq_len:
                    connect_size = cluster_size // 4
                    pattern[i:end_i, i+cluster_size:i+cluster_size+connect_size] = 1
        
        else:
            # Default: adaptive band pattern
            bandwidth = max(8, seq_len // 16)
            for i in range(seq_len):
                start = max(0, i - bandwidth // 2)
                end = min(seq_len, i + bandwidth // 2 + 1)
                pattern[i, start:end] = 1
                # Add strategic global connections
                if i % (seq_len // 8) == 0:
                    pattern[i, :] = 1
        
        return pattern
    
    def _expand_pattern_to_mask(self, pattern: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Expand 2D pattern to full attention mask."""
        if pattern.shape[-1] > seq_len:
            pattern = pattern[:seq_len, :seq_len]
        elif pattern.shape[-1] < seq_len:
            # Pad pattern if needed
            padded_pattern = torch.zeros(seq_len, seq_len)
            padded_pattern[:pattern.shape[0], :pattern.shape[1]] = pattern
            pattern = padded_pattern
        
        # Expand to batch and head dimensions
        batch_size = 1  # We'll expand later if needed
        mask = pattern.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
        mask = mask.expand(batch_size, self.num_heads, seq_len, seq_len)
        
        return mask
    
    def _collect_performance_feedback(self,
                                   pattern_gene: AttentionPatternGene,
                                   attn_weights: torch.Tensor,
                                   x: torch.Tensor,
                                   context: Dict,
                                   pattern_info: Dict):
        """Collect performance feedback for evolutionary learning."""
        # Calculate performance metrics
        performance_metrics = self._calculate_pattern_performance(
            attn_weights, x, context
        )
        
        # Determine if pattern was successful
        success = self._evaluate_pattern_success(performance_metrics)
        
        # Update gene fitness
        pattern_gene.update_fitness(success, performance_metrics)
        
        # Store feedback for evolution
        feedback_entry = {
            'pattern_id': id(pattern_gene),
            'context': context,
            'performance': performance_metrics,
            'success': success,
            'timestamp': datetime.now(),
            'pattern_type': pattern_gene.pattern_type
        }
        self.performance_feedback_buffer.append(feedback_entry)
        
        # Update adaptive parameters
        self._update_adaptive_parameters(success, performance_metrics)
    
    def _calculate_pattern_performance(self, 
                                     attn_weights: torch.Tensor,
                                     x: torch.Tensor,
                                     context: Dict) -> Dict:
        """Calculate performance metrics for attention pattern."""
        # Attention effectiveness metrics
        attn_effectiveness = attn_weights.sum(dim=-1).mean().item()
        attn_variance = attn_weights.var().item()
        
        # Efficiency metrics
        pattern_density = (attn_weights != 0).float().mean().item()
        efficiency_gain = 1.0 / max(pattern_density, 0.001)
        
        # Conceptual coherence (simplified)
        input_variance = x.var().item()
        coherence_score = 1.0 - min(attn_variance / max(input_variance, 0.001), 1.0)
        
        return {
            'attention_effectiveness': attn_effectiveness,
            'attention_variance': attn_variance,
            'pattern_density': pattern_density,
            'efficiency_gain': efficiency_gain,
            'conceptual_coherence': coherence_score,
            'confidence': min(attn_effectiveness * coherence_score, 1.0)
        }
    
    def _evaluate_pattern_success(self, performance_metrics: Dict) -> bool:
        """Evaluate if pattern usage was successful."""
        confidence = performance_metrics.get('confidence', 0.0)
        effectiveness = performance_metrics.get('attention_effectiveness', 0.0)
        coherence = performance_metrics.get('conceptual_coherence', 0.0)
        
        # Success criteria
        success_threshold = 0.6
        return (confidence > success_threshold and 
                effectiveness > 0.3 and 
                coherence > 0.4)
    
    def _update_adaptive_parameters(self, success: bool, performance_metrics: Dict):
        """Update adaptive learning parameters based on performance."""
        confidence = performance_metrics.get('confidence', 0.0)
        
        # Adjust exploration rate
        if success and confidence > 0.8:
            # Reduce exploration when performing well
            self.adaptive_params['current_exploration_rate'] *= 0.95
        elif not success or confidence < 0.4:
            # Increase exploration when struggling
            self.adaptive_params['current_exploration_rate'] = min(
                self.adaptive_params['current_exploration_rate'] * 1.2,
                self.exploration_rate * 2.0  # Cap at 2x base rate
            )
        
        # Track pattern stagnation
        if success:
            self.adaptive_params['pattern_stagnation_count'] = 0
            if confidence > 0.9:
                self.adaptive_params['last_major_improvement'] = datetime.now()
        else:
            self.adaptive_params['pattern_stagnation_count'] += 1
        
        # Calculate evolution urgency
        time_since_improvement = (datetime.now() - 
                                self.adaptive_params['last_major_improvement']).total_seconds()
        stagnation_urgency = min(self.adaptive_params['pattern_stagnation_count'] / 50, 1.0)
        time_urgency = min(time_since_improvement / 3600, 1.0)  # 1 hour max
        
        self.adaptive_params['evolution_urgency'] = max(stagnation_urgency, time_urgency)
    
    def _trigger_evolution(self, context: Dict, seq_len: int):
        """Trigger evolutionary optimization."""
        print(f"[Evolution] Triggering generation {self.evolution_engine.generation + 1}")
        
        # Evolve population
        self.evolution_engine.evolve_generation(seq_len, context)
        
        # Adaptive evolution parameters based on urgency
        urgency = self.adaptive_params['evolution_urgency']
        if urgency > 0.7:
            # Aggressive evolution
            self.evolution_engine.mutation_rate = min(0.4, self.evolution_engine.mutation_rate * 1.5)
            self.evolution_engine.crossover_rate = min(0.6, self.evolution_engine.crossover_rate * 1.3)
        elif urgency < 0.3:
            # Conservative evolution
            self.evolution_engine.mutation_rate = max(0.05, self.evolution_engine.mutation_rate * 0.7)
        
        # Report evolution progress
        evolution_report = self.evolution_engine.get_evolution_report()
        print(f"[Evolution] Generation {evolution_report['generation']}: "
              f"Best fitness = {evolution_report['best_fitness']:.3f}")
    
    def _generate_context_key(self, context: Dict, seq_len: int) -> str:
        """Generate key for context pattern mapping."""
        if not context:
            return f"generic_{seq_len}"
        
        domain = context.get('domain', 'unknown')
        modality = context.get('modality', 'text')
        task_type = context.get('task_type', 'general')
        
        return f"{domain}_{modality}_{task_type}_{seq_len}"
    
    def get_evolution_report(self) -> Dict:
        """Get comprehensive evolution report."""
        evolution_report = self.evolution_engine.get_evolution_report()
        ckg_report = self.get_ckg_performance_report()
        
        # Pattern diversity analysis
        pattern_types = [gene.pattern_type for gene in self.evolution_engine.population]
        type_counts = {pt: pattern_types.count(pt) for pt in set(pattern_types)}
        
        return {
            **evolution_report,
            **ckg_report,
            'optimization_steps': self.optimization_steps,
            'context_pattern_cache_size': len(self.context_pattern_map),
            'pattern_type_diversity': type_counts,
            'adaptive_parameters': self.adaptive_params,
            'performance_feedback_entries': len(self.performance_feedback_buffer),
            'current_exploration_rate': self.adaptive_params['current_exploration_rate'],
            'evolution_urgency': self.adaptive_params['evolution_urgency']
        }
    
    def export_learned_patterns(self, filepath: str):
        """Export learned patterns for transfer learning."""
        patterns_data = {
            'evolution_engine': {
                'generation': self.evolution_engine.generation,
                'best_fitness_history': self.evolution_engine.best_fitness_history,
                'novelty_archive_size': len(self.evolution_engine.novelty_archive)
            },
            'context_pattern_map': {
                context_key: {
                    'pattern_type': gene.pattern_type,
                    'fitness': gene.fitness,
                    'usage_count': gene.usage_count,
                    'success_rate': gene.success_count / max(gene.usage_count, 1)
                }
                for context_key, gene in self.context_pattern_map.items()
            },
            'adaptive_parameters': self.adaptive_params,
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(patterns_data, f, indent=2)
        
        print(f"Learned patterns exported to {filepath}")
    
    def import_learned_patterns(self, filepath: str):
        """Import learned patterns (simplified - would need pattern matrix reconstruction)."""
        try:
            with open(filepath, 'r') as f:
                patterns_data = json.load(f)
            
            # Would reconstruct patterns from saved data
            print(f"Pattern import functionality would reconstruct patterns from {filepath}")
            
        except Exception as e:
            print(f"Pattern import failed: {e}")