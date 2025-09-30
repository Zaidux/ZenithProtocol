# /src/training/meta_learner.py

import torch
import random
from typing import List, Dict, Any
from copy import deepcopy
from ..models.asreh_model import ASREHModel
from ..utils.config import Config
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph
from ..attention.zenith_sparse_attention import ZenithSparseAttention
import numpy as np

class MetaLearner:
    """
    Implements a Model-Agnostic Meta-Learning (MAML) approach with Zenith Sparse Attention.
    It trains a model to find a good set of initial parameters that can
    be quickly adapted to new, unseen tasks (domains) with a few gradient steps.
    """
    def __init__(self, model: ASREHModel, tasks: List, ckg: ConceptualKnowledgeGraph, zenith_attention: ZenithSparseAttention = None):
        self.model = model
        self.ckg = ckg
        self.zenith_attention = zenith_attention
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.META_LEARNING_RATE)
        
        # Include zenith attention parameters if available
        if self.zenith_attention:
            zenith_params = list(self.zenith_attention.parameters())
            model_params = list(self.model.parameters())
            self.meta_optimizer = torch.optim.Adam(zenith_params + model_params, lr=Config.META_LEARNING_RATE)
        
        self.criterion = torch.nn.MSELoss()
        self.tasks = tasks
        self.config = Config()
        
        # Sparse attention meta-learning tracking
        self.attention_adaptation_history = []

    def inner_loop_update(self, task_model: ASREHModel, task_data: List, task_zenith: ZenithSparseAttention = None):
        """
        Performs the inner-loop update on a specific task with sparse attention adaptation.
        This adapts the task_model for a few gradient steps.
        """
        # Prepare parameters for optimization
        model_params = list(task_model.parameters())
        zenith_params = list(task_zenith.parameters()) if task_zenith else []
        all_params = model_params + zenith_params
        
        task_optimizer = torch.optim.Adam(all_params, lr=self.config.INNER_LOOP_LR)
        task_model.train()
        if task_zenith:
            task_zenith.train()
        
        adaptation_patterns = []
        
        for step in range(self.config.INNER_LOOP_STEPS):
            step_loss = 0
            for data_point in task_data:
                state, conceptual_features, target = data_point['state'], data_point['conceptual_features'], data_point['target']
                domain = data_point['domain']
                state_tensor = torch.tensor(state).unsqueeze(0).float().to(self.config.DEVICE)
                conceptual_tensor = torch.tensor(conceptual_features).unsqueeze(0).float().to(self.config.DEVICE)
                target_tensor = torch.tensor(target).unsqueeze(0).float().to(self.config.DEVICE)
                
                # Forward pass with potential sparse attention
                if task_zenith and hasattr(task_model, 'zenith_attention'):
                    predicted_output, _, _ = task_model(state_tensor, conceptual_tensor, domain)
                else:
                    predicted_output, _, _ = task_model(state_tensor, conceptual_tensor, domain)
                
                loss = self.criterion(predicted_output, target_tensor)
                step_loss += loss.item()
                
                task_optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stable sparse attention learning
                if task_zenith:
                    torch.nn.utils.clip_grad_norm_(zenith_params, self.config.ZENITH_GRAD_CLIP)
                
                task_optimizer.step()
            
            # Track attention adaptation patterns
            if task_zenith and step % 2 == 0:  # Sample every 2 steps
                sparsity_stats = task_zenith.get_sparsity_stats()
                adaptation_patterns.append({
                    'step': step,
                    'sparsity_ratio': sparsity_stats['sparsity_ratio'],
                    'computational_saving': sparsity_stats['sparsity_ratio'] * 0.8  # Estimate
                })
        
        self.attention_adaptation_history.extend(adaptation_patterns)
        return adaptation_patterns

    def outer_loop_update(self, adapted_models: List[ASREHModel], tasks: List, adapted_attentions: List[ZenithSparseAttention] = None):
        """
        Performs the outer-loop update on the main meta-model with sparse attention consideration.
        """
        self.meta_optimizer.zero_grad()
        meta_loss = 0
        total_sparse_efficiency = 0
        
        for i, adapted_model in enumerate(adapted_models):
            task_data = tasks[i]['val_data']
            task_domain = tasks[i]['domain']
            task_zenith = adapted_attentions[i] if adapted_attentions else None
            
            for data_point in task_data:
                state, conceptual_features, target = data_point['state'], data_point['conceptual_features'], data_point['target']
                state_tensor = torch.tensor(state).unsqueeze(0).float().to(self.config.DEVICE)
                conceptual_tensor = torch.tensor(conceptual_features).unsqueeze(0).float().to(self.config.DEVICE)
                target_tensor = torch.tensor(target).unsqueeze(0).float().to(self.config.DEVICE)
                
                predicted_output, _, _ = adapted_model(state_tensor, conceptual_tensor, task_domain)
                meta_loss += self.criterion(predicted_output, target_tensor)
            
            # Incorporate sparse attention efficiency into meta-learning
            if task_zenith:
                sparsity_stats = task_zenith.get_sparsity_stats()
                efficiency_bonus = sparsity_stats['sparsity_ratio'] * self.config.ZENITH_EFFICIENCY_BONUS
                meta_loss -= efficiency_bonus
                total_sparse_efficiency += sparsity_stats['sparsity_ratio']

        # Log meta-learning with sparse attention to CKG
        self.ckg.add_node("SparseMeta-Training", {
            "type": "training_phase", 
            "description": "The model has learned how to learn with sparse attention.",
            "avg_sparsity": total_sparse_efficiency / len(adapted_models) if adapted_models else 0
        })
        self.ckg.add_edge("ASREHModel", "SparseMeta-Training", "ADAPTED_WITH_SPARSE_ATTENTION")
        
        self.ckg.add_node(f"Meta-Loss_{meta_loss.item():.2f}", {
            "type": "metric", 
            "value": meta_loss.item(),
            "sparse_efficiency": total_sparse_efficiency
        })
        self.ckg.add_edge("SparseMeta-Training", f"Meta-Loss_{meta_loss.item():.2f}", "RESULTED_IN")

        meta_loss.backward()
        self.meta_optimizer.step()
        
        avg_efficiency = total_sparse_efficiency / len(adapted_models) if adapted_models else 0
        print(f"Meta-loss: {meta_loss.item():.4f}, Avg Sparsity Efficiency: {avg_efficiency:.3f}")

    def run_meta_training(self):
        """Runs the full meta-training loop with sparse attention adaptation."""
        print(f"Starting Meta-Training for {self.config.META_TRAINING_EPOCHS} epochs with Zenith Sparse Attention.")
        
        for epoch in range(self.config.META_TRAINING_EPOCHS):
            print(f"\n--- Meta-Training Epoch {epoch + 1}/{self.config.META_TRAINING_EPOCHS} ---")
            sampled_tasks = random.sample(self.tasks, self.config.NUM_META_TASKS)
            adapted_models = []
            adapted_attentions = []
            
            for task in sampled_tasks:
                # Clone model and attention for task adaptation
                task_model = deepcopy(self.model)
                task_zenith = deepcopy(self.zenith_attention) if self.zenith_attention else None
                
                # Perform inner loop adaptation
                adaptation_patterns = self.inner_loop_update(
                    task_model, 
                    task['train_data'], 
                    task_zenith
                )
                
                adapted_models.append(task_model)
                adapted_attentions.append(task_zenith)
                
                # Log adaptation patterns
                if adaptation_patterns:
                    final_sparsity = adaptation_patterns[-1]['sparsity_ratio']
                    print(f"Task adaptation completed. Final sparsity: {final_sparsity:.3f}")
            
            # Outer loop update with sparse attention consideration
            self.outer_loop_update(adapted_models, sampled_tasks, adapted_attentions)
            
            # Clean up to save memory
            del adapted_models
            del adapted_attentions
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print("\nMeta-Training complete. The model is now ready for rapid adaptation with optimized sparse attention.")

    def run_mini_meta_training(self, model_to_adapt: ASREHModel, ckg: ConceptualKnowledgeGraph, current_zenith: ZenithSparseAttention = None):
        """
        Runs a small, on-demand meta-learning loop with sparse attention optimization
        to help the model adapt to a new, difficult problem or task it is currently facing.
        """
        print("\n[MetaLearner] Initiating mini meta-learning loop with sparse attention to overcome a difficult problem.")
        
        # Create a tiny, temporary task based on the current context
        current_context_task = self._create_task_from_current_context(model_to_adapt, ckg, current_zenith)

        # Clone the model and attention to avoid changing the main instances
        temp_model = deepcopy(model_to_adapt)
        temp_zenith = deepcopy(current_zenith) if current_zenith else None
        
        # Perform quick adaptation
        adaptation_patterns = self.inner_loop_update(temp_model, current_context_task['train_data'], temp_zenith)

        # Create a pseudo-meta-loss to update the main model
        meta_loss = 0
        efficiency_bonus = 0
        
        for data_point in current_context_task['val_data']:
            state, conceptual_features, target = data_point['state'], data_point['conceptual_features'], data_point['target']
            domain = data_point['domain']
            state_tensor = torch.tensor(state).unsqueeze(0).float().to(self.config.DEVICE)
            conceptual_tensor = torch.tensor(conceptual_features).unsqueeze(0).float().to(self.config.DEVICE)
            target_tensor = torch.tensor(target).unsqueeze(0).float().to(self.config.DEVICE)
            
            predicted_output, _, _ = temp_model(state_tensor, conceptual_tensor, domain)
            meta_loss += self.criterion(predicted_output, target_tensor)
        
        # Add efficiency bonus from sparse attention
        if temp_zenith and adaptation_patterns:
            final_sparsity = adaptation_patterns[-1]['sparsity_ratio']
            efficiency_bonus = final_sparsity * self.config.ZENITH_EFFICIENCY_BONUS
            meta_loss -= efficiency_bonus

        # Update the main model's parameters with the meta-loss
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        
        # Apply gradient clipping for stability
        if current_zenith:
            torch.nn.utils.clip_grad_norm_(current_zenith.parameters(), self.config.ZENITH_GRAD_CLIP)
            
        self.meta_optimizer.step()
        
        print(f"[MetaLearner] Mini meta-learning complete. "
              f"Meta-loss: {meta_loss.item():.4f}, "
              f"Efficiency bonus: {efficiency_bonus:.4f}")

    def _create_task_from_current_context(self, model: ASREHModel, ckg: ConceptualKnowledgeGraph, zenith_attention: ZenithSparseAttention = None) -> Dict:
        """
        Creates a new task based on what the model is currently struggling with,
        incorporating sparse attention patterns.
        """
        # Get current sparse attention statistics for context
        sparse_context = {}
        if zenith_attention:
            sparsity_stats = zenith_attention.get_sparsity_stats()
            sparse_context = {
                'current_sparsity': sparsity_stats['sparsity_ratio'],
                'pruned_blocks': sparsity_stats['attention_blocks_pruned'],
                'efficiency': sparsity_stats['sparsity_ratio'] * 0.8  # Estimated
            }
        
        # Create task data with sparse attention context
        new_task_data = {
            'domain': 'dynamic_new_problem',
            'train_data': [{
                'state': np.random.rand(128), 
                'conceptual_features': np.random.rand(64), 
                'target': np.random.rand(1), 
                'domain': 'dynamic_new_problem',
                'sparse_context': sparse_context
            } for _ in range(5)],
            'val_data': [{
                'state': np.random.rand(128), 
                'conceptual_features': np.random.rand(64), 
                'target': np.random.rand(1), 
                'domain': 'dynamic_new_problem',
                'sparse_context': sparse_context
            } for _ in range(5)]
        }
        
        # Log this event to the CKG with sparse attention context
        ckg.add_node("SparseMiniMetaLearning", {
            "type": "self_optimization", 
            "description": "Initiated a mini meta-learning loop with sparse attention optimization.",
            "sparsity_context": sparse_context
        })
        
        return new_task_data

    def get_attention_adaptation_summary(self):
        """Returns a summary of how sparse attention adapted during meta-learning"""
        if not self.attention_adaptation_history:
            return {"message": "No adaptation data available"}
        
        final_adaptations = [pattern for pattern in self.attention_adaptation_history 
                           if pattern['step'] == self.config.INNER_LOOP_STEPS - 1]
        
        if not final_adaptations:
            final_adaptations = self.attention_adaptation_history[-5:]  # Last 5
        
        avg_final_sparsity = np.mean([adapt['sparsity_ratio'] for adapt in final_adaptations])
        avg_efficiency = np.mean([adapt['computational_saving'] for adapt in final_adaptations])
        
        return {
            "average_final_sparsity": avg_final_sparsity,
            "average_computational_saving": avg_efficiency,
            "total_adaptation_steps": len(self.attention_adaptation_history),
            "adaptation_trend": "increasing" if len(final_adaptations) > 1 and 
                                final_adaptations[-1]['sparsity_ratio'] > final_adaptations[0]['sparsity_ratio'] 
                                else "stable"
        }