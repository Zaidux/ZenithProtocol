"""
Sparse Training - Only updates relevant parameters during training
"""

import torch
import torch.nn as nn
import torch.nn.utils as utils
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np

class SparsityType(Enum):
    STRUCTURED = "structured"
    UNSTRUCTURED = "unstructured"
    BLOCK = "block"
    CHANNEL = "channel"

@dataclass
class SparsityConfig:
    sparsity_type: SparsityType
    sparsity_level: float  # 0.0 to 1.0
    update_frequency: int = 100  # How often to update sparsity pattern
    gradual_increase: bool = True
    target_layers: List[str] = None

class SparseTraining:
    def __init__(self, model: nn.Module):
        self.model = model
        self.sparsity_masks = {}
        self.training_stats = {}
        self.parameter_importance = {}
        
    def setup_sparse_training(self, config: SparsityConfig) -> nn.Module:
        """Setup model for sparse training"""
        
        print(f"🎯 Setting up {config.sparsity_type.value} sparse training...")
        print(f"   Target sparsity: {config.sparsity_level:.1%}")
        
        # Initialize sparsity masks
        self._initialize_sparsity_masks(config)
        
        # Create sparse training wrapper
        sparse_model = self._create_sparse_wrapper(config)
        
        self.training_stats = {
            'config': config,
            'total_parameters': self._count_parameters(),
            'active_parameters': self._count_active_parameters(),
            'sparsity_achieved': 0.0,
            'training_iterations': 0
        }
        
        return sparse_model
    
    def _initialize_sparsity_masks(self, config: SparsityConfig):
        """Initialize sparsity masks for model parameters"""
        
        self.sparsity_masks.clear()
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Only apply to target layers if specified
                if config.target_layers and name not in config.target_layers:
                    continue
                
                # Create initial mask (all ones - no sparsity initially)
                mask = torch.ones_like(param.data, dtype=torch.bool)
                self.sparsity_masks[name] = mask
                
                # Initialize parameter importance tracking
                self.parameter_importance[name] = torch.zeros_like(param.data)
    
    def _create_sparse_wrapper(self, config: SparsityConfig) -> nn.Module:
        """Create a wrapper that applies sparsity during training"""
        
        class SparseTrainingWrapper(nn.Module):
            def __init__(self, original_model, sparsity_manager, config):
                super().__init__()
                self.original_model = original_model
                self.sparsity_manager = sparsity_manager
                self.config = config
                self.iteration = 0
            
            def forward(self, *args, **kwargs):
                return self.original_model(*args, **kwargs)
            
            def apply_sparsity(self):
                """Apply sparsity masks to parameters"""
                self.sparsity_manager._apply_masks()
            
            def update_sparsity_pattern(self, gradients):
                """Update sparsity pattern based on gradient information"""
                self.iteration += 1
                
                if self.iteration % self.config.update_frequency == 0:
                    self.sparsity_manager._update_masks(gradients, self.config)
        
        return SparseTrainingWrapper(self.model, self, config)
    
    def _apply_masks(self):
        """Apply sparsity masks to model parameters"""
        for name, param in self.model.named_parameters():
            if name in self.sparsity_masks:
                mask = self.sparsity_masks[name]
                param.data *= mask  # Zero out masked parameters
    
    def _update_masks(self, gradients: Dict[str, torch.Tensor], config: SparsityConfig):
        """Update sparsity masks based on gradient information"""
        
        print("   Updating sparsity pattern...")
        
        # Update parameter importance based on gradients
        self._update_parameter_importance(gradients)
        
        # Calculate target number of active parameters
        for name, param in self.model.named_parameters():
            if name in self.sparsity_masks:
                total_params = param.data.numel()
                target_active = int(total_params * (1 - config.sparsity_level))
                
                # Get most important parameters
                importance_scores = self.parameter_importance[name].abs()
                _, important_indices = torch.topk(
                    importance_scores.flatten(), 
                    target_active
                )
                
                # Create new mask
                new_mask = torch.zeros_like(param.data, dtype=torch.bool)
                new_mask.flatten()[important_indices] = True
                
                self.sparsity_masks[name] = new_mask
        
        # Update statistics
        self._update_training_stats()
    
    def _update_parameter_importance(self, gradients: Dict[str, torch.Tensor]):
        """Update parameter importance scores based on gradients"""
        
        for name, param in self.model.named_parameters():
            if name in gradients and name in self.parameter_importance:
                grad = gradients[name]
                
                # Update importance using exponential moving average
                importance = self.parameter_importance[name]
                new_importance = grad.abs()  # Use gradient magnitude as importance
                
                # EMA update
                alpha = 0.9
                self.parameter_importance[name] = (
                    alpha * importance + (1 - alpha) * new_importance
                )
    
    def gradual_sparsity_increase(self, model: nn.Module, 
                                final_sparsity: float,
                                steps: int = 10) -> nn.Module:
        """Gradually increase sparsity during training"""
        
        print(f"📈 Setting up gradual sparsity increase to {final_sparsity:.1%}")
        
        current_sparsity = 0.0
        sparsity_increment = final_sparsity / steps
        
        config = SparsityConfig(
            sparsity_type=SparsityType.UNSTRUCTURED,
            sparsity_level=current_sparsity,
            gradual_increase=True
        )
        
        sparse_model = self.setup_sparse_training(config)
        
        # Add callback for sparsity increase
        def sparsity_increase_callback(iteration):
            nonlocal current_sparsity
            if iteration % 100 == 0 and current_sparsity < final_sparsity:
                current_sparsity += sparsity_increment
                print(f"   Increasing sparsity to {current_sparsity:.1%}")
                # Update masks with new sparsity level
                # This would be called during training
        
        return sparse_model
    
    def layer_wise_sparsity(self, model: nn.Module,
                          sparsity_pattern: Dict[str, float]) -> nn.Module:
        """Apply different sparsity levels to different layers"""
        
        print("🎛️ Applying layer-wise sparsity...")
        
        config = SparsityConfig(
            sparsity_type=SparsityType.UNSTRUCTURED,
            sparsity_level=0.5,  # Default
            target_layers=list(sparsity_pattern.keys())
        )
        
        sparse_model = self.setup_sparse_training(config)
        
        # Set individual sparsity levels
        for layer_name, sparsity in sparsity_pattern.items():
            if layer_name in self.sparsity_masks:
                # This would be applied during mask updates
                print(f"   {layer_name}: {sparsity:.1%} sparsity")
        
        return sparse_model
    
    def dynamic_sparsity_adjustment(self, model: nn.Module,
                                  performance_metric: Callable,
                                  target_performance: float) -> nn.Module:
        """Dynamically adjust sparsity based on performance"""
        
        print("🎚️ Setting up dynamic sparsity adjustment...")
        
        config = SparsityConfig(
            sparsity_type=SparsityType.UNSTRUCTURED,
            sparsity_level=0.3  # Initial sparsity
        )
        
        sparse_model = self.setup_sparse_training(config)
        
        # This would be called during training to adjust sparsity
        def performance_based_adjustment(current_performance):
            current_sparsity = config.sparsity_level
            
            if current_performance > target_performance * 1.1:
                # Performance is good, can increase sparsity
                new_sparsity = min(current_sparsity + 0.05, 0.8)
                if new_sparsity != current_sparsity:
                    print(f"   Increasing sparsity to {new_sparsity:.1%}")
                    config.sparsity_level = new_sparsity
            elif current_performance < target_performance * 0.9:
                # Performance is poor, decrease sparsity
                new_sparsity = max(current_sparsity - 0.05, 0.1)
                if new_sparsity != current_sparsity:
                    print(f"   Decreasing sparsity to {new_sparsity:.1%}")
                    config.sparsity_level = new_sparsity
        
        return sparse_model
    
    def get_sparsity_stats(self) -> Dict[str, Any]:
        """Get current sparsity statistics"""
        
        total_params = 0
        active_params = 0
        
        for name, mask in self.sparsity_masks.items():
            layer_params = mask.numel()
            layer_active = mask.sum().item()
            
            total_params += layer_params
            active_params += layer_active
        
        achieved_sparsity = 1 - (active_params / total_params) if total_params > 0 else 0
        
        stats = self.training_stats.copy()
        stats.update({
            'achieved_sparsity': achieved_sparsity,
            'active_parameters': active_params,
            'total_parameters': total_params,
            'compression_ratio': total_params / active_params if active_params > 0 else 1.0
        })
        
        return stats
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _count_active_parameters(self) -> int:
        """Count currently active parameters"""
        active_count = 0
        for name, mask in self.sparsity_masks.items():
            active_count += mask.sum().item()
        return int(active_count)
    
    def _update_training_stats(self):
        """Update training statistics"""
        stats = self.get_sparsity_stats()
        self.training_stats.update(stats)
