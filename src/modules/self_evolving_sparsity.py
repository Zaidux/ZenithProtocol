# /src/modules/self_evolving_sparsity.py

"""
Self-Evolving Sparse Attention with Pattern Discovery and Optimization
======================================================================
Automatically discovers, evaluates, and evolves optimal sparse attention patterns
based on performance feedback and conceptual understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
from collections import defaultdict, deque
import random
import json
from datetime import datetime

from .ckg_guided_sparse_attention import CKGSparseAttention
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph
from ..models.hyper_conceptual_thinking import ConceptDiscoveryEngine

class AttentionPatternGene:
    """
    Represents a sparse attention pattern as an evolvable gene.
    """
    def __init__(self, 
                 pattern_matrix: torch.Tensor,
                 pattern_type: str,
                 discovery_context: Dict,
                 fitness: float = 0.0):
        self.pattern_matrix = pattern_matrix
        self.pattern_type = pattern_type
        self.discovery_context = discovery_context
        self.fitness = fitness
        self.usage_count = 0
        self.success_count = 0
        self.discovery_time = datetime.now()
        self.last_used = None
        self.mutation_history = []
        
        # Pattern metadata
        self.sparsity = (pattern_matrix == 0).float().mean().item()
        self.sequence_length = pattern_matrix.shape[-1]
        self.complexity_score = self._calculate_complexity()
    
    def _calculate_complexity(self) -> float:
        """Calculate pattern complexity score."""
        # Patterns with more structure are less complex than random patterns
        if self.pattern_matrix.sum() == 0:
            return 0.0
        
        # Calculate entropy of the pattern
        unique, counts = torch.unique(self.pattern_matrix, return_counts=True)
        probabilities = counts.float() / counts.sum()
        entropy = -torch.sum(probabilities * torch.log2(probabilities + 1e-8))
        
        # Normalize to 0-1 range
        max_entropy = math.log2(self.pattern_matrix.numel())
        return (entropy / max_entropy).item()
    
    def mutate(self, mutation_rate: float = 0.1) -> 'AttentionPatternGene':
        """Create a mutated version of this pattern."""
        mutated_matrix = self.pattern_matrix.clone()
        seq_len = mutated_matrix.shape[-1]
        
        # Different mutation strategies
        mutation_type = random.choice(['flip', 'shift', 'scale', 'recombine'])
        
        if mutation_type == 'flip':
            # Randomly flip some attention connections
            flip_mask = torch.rand_like(mutated_matrix) < mutation_rate
            mutated_matrix[flip_mask] = 1 - mutated_matrix[flip_mask]
            
        elif mutation_type == 'shift':
            # Shift attention windows
            shift_amount = random.randint(-2, 2)
            if shift_amount != 0:
                shifted = torch.roll(mutated_matrix, shifts=shift_amount, dims=-1)
                # Preserve causal masking
                if shift_amount > 0:
                    shifted[..., :shift_amount] = 0
                else:
                    shifted[..., shift_amount:] = 0
                mutated_matrix = shifted
        
        elif mutation_type == 'scale':
            # Scale the attention density
            current_density = mutated_matrix.float().mean()
            target_density = current_density * random.uniform(0.7, 1.3)
            target_density = max(0.1, min(0.9, target_density))
            
            if target_density > current_density:
                # Add connections
                zero_positions = (mutated_matrix == 0).nonzero(as_tuple=True)
                num_to_add = int((target_density - current_density) * mutated_matrix.numel())
                if len(zero_positions[0]) > 0 and num_to_add > 0:
                    indices = random.sample(range(len(zero_positions[0])), 
                                          min(num_to_add, len(zero_positions[0])))
                    for idx in indices:
                        mutated_matrix[zero_positions[0][idx], 
                                     zero_positions[1][idx]] = 1
            else:
                # Remove connections
                one_positions = (mutated_matrix == 1).nonzero(as_tuple=True)
                num_to_remove = int((current_density - target_density) * mutated_matrix.numel())
                if len(one_positions[0]) > 0 and num_to_remove > 0:
                    indices = random.sample(range(len(one_positions[0])), 
                                          min(num_to_remove, len(one_positions[0])))
                    for idx in indices:
                        mutated_matrix[one_positions[0][idx], 
                                     one_positions[1][idx]] = 0
        
        mutated_gene = AttentionPatternGene(
            mutated_matrix, 
            f"mutated_{self.pattern_type}",
            self.discovery_context,
            self.fitness * 0.9  # Slight fitness decay for mutations
        )
        mutated_gene.mutation_history = self.mutation_history + [mutation_type]
        
        return mutated_gene
    
    def crossover(self, other: 'AttentionPatternGene') -> 'AttentionPatternGene':
        """Crossover two patterns to create a new one."""
        if self.pattern_matrix.shape != other.pattern_matrix.shape:
            return self  # Cannot crossover different shapes
        
        # Create mask for crossover
        crossover_mask = torch.rand_like(self.pattern_matrix.float()) < 0.5
        child_matrix = torch.where(crossover_mask, 
                                 self.pattern_matrix, 
                                 other.pattern_matrix)
        
        child_gene = AttentionPatternGene(
            child_matrix,
            f"crossover_{self.pattern_type}_{other.pattern_type}",
            {**self.discovery_context, **other.discovery_context},
            (self.fitness + other.fitness) / 2
        )
        
        return child_gene
    
    def update_fitness(self, success: bool, performance_metrics: Dict):
        """Update fitness based on usage success."""
        self.usage_count += 1
        if success:
            self.success_count += 1
        
        # Calculate new fitness
        success_rate = self.success_count / max(self.usage_count, 1)
        
        # Factor in performance metrics
        performance_bonus = 0.0
        if 'confidence' in performance_metrics:
            performance_bonus = performance_metrics['confidence'] * 0.5
        if 'efficiency_gain' in performance_metrics:
            performance_bonus += min(performance_metrics['efficiency_gain'] * 0.1, 0.3)
        
        # Complexity penalty - prefer simpler patterns
        complexity_penalty = self.complexity_score * 0.2
        
        self.fitness = success_rate + performance_bonus - complexity_penalty
        self.last_used = datetime.now()
        
        return self.fitness

class PatternEvolutionEngine:
    """
    Evolutionary algorithm for attention pattern optimization.
    """
    
    def __init__(self,
                 population_size: int = 50,
                 elite_count: int = 5,
                 mutation_rate: float = 0.15,
                 crossover_rate: float = 0.3,
                 novelty_threshold: float = 0.7):
        
        self.population_size = population_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.novelty_threshold = novelty_threshold
        
        self.population: List[AttentionPatternGene] = []
        self.generation = 0
        self.best_fitness_history = []
        self.novelty_archive = set()
        
        # Pattern templates for different scenarios
        self.base_templates = self._initialize_base_templates()
    
    def _initialize_base_templates(self) -> Dict[str, torch.Tensor]:
        """Initialize base attention pattern templates."""
        templates = {}
        
        # Local window patterns
        for window_size in [8, 16, 32, 64]:
            pattern = self._create_local_pattern(128, window_size)
            templates[f'local_{window_size}'] = pattern
        
        # Strided patterns
        for stride in [2, 4, 8]:
            pattern = self._create_strided_pattern(128, stride)
            templates[f'strided_{stride}'] = pattern
        
        # Global + local patterns
        pattern = self._create_global_local_pattern(128, local_size=16)
        templates['global_local'] = pattern
        
        return templates
    
    def _create_local_pattern(self, seq_len: int, window_size: int) -> torch.Tensor:
        """Create local window attention pattern."""
        pattern = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            pattern[i, start:end] = 1
        return pattern
    
    def _create_strided_pattern(self, seq_len: int, stride: int) -> torch.Tensor:
        """Create strided attention pattern."""
        pattern = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            pattern[i, ::stride] = 1
            pattern[i, i] = 1  # Always attend to self
        return pattern
    
    def _create_global_local_pattern(self, seq_len: int, local_size: int) -> torch.Tensor:
        """Create global + local attention pattern."""
        pattern = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            # Local attention
            start = max(0, i - local_size // 2)
            end = min(seq_len, i + local_size // 2 + 1)
            pattern[i, start:end] = 1
            
            # Global attention to key positions
            if i % 8 == 0:  # Key tokens get global attention
                pattern[i, :] = 1
            if i < 4:  # First few tokens are global
                pattern[i, :] = 1
        return pattern
    
    def initialize_population(self, seq_len: int, context: Dict):
        """Initialize population with diverse patterns."""
        self.population.clear()
        
        # Add base templates
        for name, template in self.base_templates.items():
            if template.shape[-1] >= seq_len:
                # Resize template if needed
                resized_template = template[:seq_len, :seq_len]
                gene = AttentionPatternGene(
                    resized_template, 
                    f"template_{name}",
                    context
                )
                self.population.append(gene)
        
        # Add random patterns for diversity
        while len(self.population) < self.population_size:
            random_pattern = self._generate_random_pattern(seq_len, context)
            if self._is_novel_pattern(random_pattern):
                gene = AttentionPatternGene(
                    random_pattern,
                    "random",
                    context
                )
                self.population.append(gene)
        
        self.generation = 1
    
    def _generate_random_pattern(self, seq_len: int, context: Dict) -> torch.Tensor:
        """Generate a random but structured attention pattern."""
        pattern = torch.zeros(seq_len, seq_len)
        
        # Different random pattern strategies
        strategy = random.choice(['banded', 'block', 'sparse_random', 'hierarchical'])
        
        if strategy == 'banded':
            # Banded random pattern
            bandwidth = random.randint(4, 32)
            for i in range(seq_len):
                start = max(0, i - bandwidth // 2)
                end = min(seq_len, i + bandwidth // 2 + 1)
                pattern[i, start:end] = 1
                # Add some random connections
                random_indices = torch.randperm(seq_len)[:random.randint(1, 4)]
                pattern[i, random_indices] = 1
                
        elif strategy == 'block':
            # Block diagonal pattern
            block_size = random.randint(8, 32)
            for i in range(0, seq_len, block_size):
                end_i = min(i + block_size, seq_len)
                pattern[i:end_i, i:end_i] = 1
                
        elif strategy == 'sparse_random':
            # Sparse random connections
            density = random.uniform(0.1, 0.4)
            mask = torch.rand(seq_len, seq_len) < density
            pattern[mask] = 1
            # Ensure self-attention
            pattern[range(seq_len), range(seq_len)] = 1
            
        elif strategy == 'hierarchical':
            # Hierarchical attention pattern
            for i in range(seq_len):
                # Local attention
                local_size = random.randint(4, 16)
                start = max(0, i - local_size // 2)
                end = min(seq_len, i + local_size // 2 + 1)
                pattern[i, start:end] = 1
                
                # Attention to hierarchy levels
                if i % 4 == 0:  # Level 1
                    pattern[i, ::4] = 1
                if i % 16 == 0:  # Level 2
                    pattern[i, ::16] = 1
        
        return pattern
    
    def _is_novel_pattern(self, pattern: torch.Tensor) -> bool:
        """Check if pattern is novel compared to archive."""
        if len(self.novelty_archive) == 0:
            return True
        
        # Simple novelty check based on pattern hash
        pattern_hash = hash(pattern.numpy().tobytes())
        if pattern_hash in self.novelty_archive:
            return False
        
        # More sophisticated similarity check
        max_similarity = 0.0
        for archived_pattern in list(self.novelty_archive)[-100:]:  # Recent patterns
            similarity = F.cosine_similarity(
                pattern.flatten().float(),
                torch.tensor(archived_pattern).flatten().float(),
                dim=0
            ).item()
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity < self.novelty_threshold
    
    def evolve_generation(self, seq_len: int, context: Dict):
        """Evolve to next generation."""
        if len(self.population) == 0:
            self.initialize_population(seq_len, context)
            return
        
        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_fitness_history.append(self.population[0].fitness)
        
        # Create new generation
        new_population = []
        
        # Elitism: keep best performers
        new_population.extend(self.population[:self.elite_count])
        
        # Crossover
        num_crossover = int(self.population_size * self.crossover_rate)
        for _ in range(num_crossover):
            parent1 = self._select_parent()
            parent2 = self._select_parent()
            if parent1 != parent2:
                child = parent1.crossover(parent2)
                if self._is_novel_pattern(child.pattern_matrix):
                    new_population.append(child)
        
        # Mutation
        num_mutation = self.population_size - len(new_population)
        for _ in range(num_mutation):
            parent = self._select_parent()
            mutated = parent.mutate(self.mutation_rate)
            if self._is_novel_pattern(mutated.pattern_matrix):
                new_population.append(mutated)
        
        # Fill remaining with new random patterns if needed
        while len(new_population) < self.population_size:
            random_pattern = self._generate_random_pattern(seq_len, context)
            if self._is_novel_pattern(random_pattern):
                gene = AttentionPatternGene(
                    random_pattern,
                    "random_new",
                    context
                )
                new_population.append(gene)
        
        self.population = new_population[:self.population_size]
        self.generation += 1
        
        # Update novelty archive
        for gene in self.population:
            pattern_hash = hash(gene.pattern_matrix.numpy().tobytes())
            self.novelty_archive.add(pattern_hash)
    
    def _select_parent(self) -> AttentionPatternGene:
        """Select parent using fitness-proportional selection."""
        fitnesses = [gene.fitness for gene in self.population]
        total_fitness = sum(fitnesses)
        
        if total_fitness == 0:
            return random.choice(self.population)
        
        probabilities = [f / total_fitness for f in fitnesses]
        return random.choices(self.population, weights=probabilities)[0]
    
    def get_best_pattern(self, seq_len: int, context: Dict) -> AttentionPatternGene:
        """Get the best pattern for current context."""
        if len(self.population) == 0:
            self.initialize_population(seq_len, context)
        
        # Filter patterns by sequence length compatibility
        compatible_patterns = [
            gene for gene in self.population 
            if gene.pattern_matrix.shape[-1] >= seq_len
        ]
        
        if not compatible_patterns:
            # Create new pattern for this sequence length
            new_pattern = self._generate_random_pattern(seq_len, context)
            new_gene = AttentionPatternGene(new_pattern, "new_context", context)
            compatible_patterns = [new_gene]
        
        # Return best compatible pattern
        return max(compatible_patterns, key=lambda x: x.fitness)
    
    def get_evolution_report(self) -> Dict:
        """Get evolution progress report."""
        if not self.population:
            return {"status": "population_not_initialized"}
        
        avg_fitness = sum(gene.fitness for gene in self.population) / len(self.population)
        max_fitness = max(gene.fitness for gene in self.population)
        min_fitness = min(gene.fitness for gene in self.population)
        
        return {
            "generation": self.generation,
            "population_size": len(self.population),
            "average_fitness": avg_fitness,
            "best_fitness": max_fitness,
            "worst_fitness": min_fitness,
            "fitness_history": self.best_fitness_history[-10:],  # Last 10 generations
            "novelty_archive_size": len(self.novelty_archive),
            "pattern_diversity": len(set(gene.pattern_type for gene in self.population))
        }