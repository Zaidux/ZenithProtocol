"""
Sparse Backpropagation - Only backpropagate through relevant paths
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np

class BackpropSparsityMethod(Enum):
    GRADIENT_MAGNITUDE = "gradient_magnitude"
    ACTIVATION_BASED = "activation_based"
    RANDOM = "random"
    ADAPTIVE = "adaptive"

@dataclass
class SparseBackpropConfig:
    method: BackpropSparsityMethod
    sparsity_level: float  # 0.0 to 1.0
    min_gradient_threshold: float = 1e-6
    update_frequency: int = 50
    layer_specific: bool = True

class SparseBackpropagation:
    def __init__(self, model: nn.Module):
        self.model = model
        self.backprop_masks = {}
        self.gradient_stats = {}
        self.config = None
        
    def enable_sparse_backprop(self, config: SparseBackpropConfig):
        """Enable sparse backpropagation for the model"""
        
        print(f"⚡ Enabling {config.method.value} sparse backpropagation...")
        print(f"   Target sparsity: {config.sparsity_level:.1%}")
        
        self.config = config
        self._initialize_backprop_masks()
        
        # Register backward hooks
        self._register_backward_hooks()
        
        self.gradient_stats = {
            'total_backward_passes': 0,
            'average_sparsity_achieved': 0.0,
            'gradient_computation_savings': 0.0,
            'layer_stats': {}
        }
    
    def _initialize_backprop_masks(self):
        """Initialize backpropagation masks"""
        
        self.backprop_masks.clear()
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Initialize mask for this layer
                self.backprop_masks[name] = None
                self.gradient_stats['layer_stats'][name] = {
                    'total_gradients': 0,
                    'pruned_gradients': 0,
                    'average_magnitude': 0.0
                }
    
    def _register_backward_hooks(self):
        """Register backward hooks for sparse backpropagation"""
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                
                def make_hook(layer_name):
                    def backward_hook(module, grad_input, grad_output):
                        return self._sparse_backward_hook(layer_name, grad_input, grad_output)
                    return backward_hook
                
                module.register_full_backward_hook(make_hook(name))
    
    def _sparse_backward_hook(self, layer_name: str, grad_input, grad_output):
        """Apply sparsity to gradients during backward pass"""
        
        if grad_output[0] is None:
            return None
        
        self.gradient_stats['total_backward_passes'] += 1
        
        # Get the output gradient
        grad = grad_output[0]
        
        # Apply sparsity based on method
        if self.config.method == BackpropSparsityMethod.GRADIENT_MAGNITUDE:
            sparse_grad = self._gradient_magnitude_pruning(grad, layer_name)
        elif self.config.method == BackpropSparsityMethod.ACTIVATION_BASED:
            sparse_grad = self._activation_based_pruning(grad, layer_name)
        elif self.config.method == BackpropSparsityMethod.RANDOM:
            sparse_grad = self._random_pruning(grad)
        elif self.config.method == BackpropSparsityMethod.ADAPTIVE:
            sparse_grad = self._adaptive_pruning(grad, layer_name)
        else:
            sparse_grad = grad
        
        # Update statistics
        self._update_gradient_stats(layer_name, grad, sparse_grad)
        
        return (sparse_grad,) + grad_output[1:]
    
    def _gradient_magnitude_pruning(self, grad: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Prune gradients based on magnitude"""
        
        if self.config.sparsity_level == 0:
            return grad
        
        # Calculate threshold based on sparsity level
        grad_flat = grad.abs().flatten()
        k = int(grad_flat.numel() * self.config.sparsity_level)
        
        if k > 0:
            # Find k-th smallest value (will be our threshold)
            threshold = torch.kthvalue(grad_flat, k).values.item()
        else:
            threshold = 0.0
        
        # Apply threshold
        mask = grad.abs() > max(threshold, self.config.min_gradient_threshold)
        sparse_grad = grad * mask
        
        return sparse_grad
    
    def _activation_based_pruning(self, grad: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Prune gradients based on activation patterns"""
        
        # This would use stored activation information
        # For now, use gradient magnitude as fallback
        return self._gradient_magnitude_pruning(grad, layer_name)
    
    def _random_pruning(self, grad: torch.Tensor) -> torch.Tensor:
        """Randomly prune gradients"""
        
        if self.config.sparsity_level == 0:
            return grad
        
        # Create random mask
        mask = torch.rand_like(grad) > self.config.sparsity_level
        sparse_grad = grad * mask
        
        return sparse_grad
    
    def _adaptive_pruning(self, grad: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Adaptively prune gradients based on layer importance"""
        
        # Simple adaptive strategy: vary sparsity by layer type
        layer_sparsity = self.config.sparsity_level
        
        # Adjust sparsity based on layer type
        if 'attention' in layer_name.lower():
            # Be more conservative with attention layers
            layer_sparsity = self.config.sparsity_level * 0.5
        elif 'output' in layer_name.lower():
            # Be more conservative with output layers
            layer_sparsity = self.config.sparsity_level * 0.7
        
        # Apply magnitude pruning with adjusted sparsity
        temp_config = SparseBackpropConfig(
            method=BackpropSparsityMethod.GRADIENT_MAGNITUDE,
            sparsity_level=layer_sparsity,
            min_gradient_threshold=self.config.min_gradient_threshold
        )
        
        # Create temporary instance to use the method
        temp_sparse = SparseBackpropagation(self.model)
        return temp_sparse._gradient_magnitude_pruning(grad, layer_name)
    
    def _update_gradient_stats(self, layer_name: str, original_grad: torch.Tensor, 
                             sparse_grad: torch.Tensor):
        """Update gradient statistics"""
        
        original_nonzero = (original_grad != 0).sum().item()
        sparse_nonzero = (sparse_grad != 0).sum().item()
        total_elements = original_grad.numel()
        
        if original_nonzero > 0:
            sparsity_achieved = 1 - (sparse_nonzero / original_nonzero)
            computation_saving = 1 - (sparse_nonzero / total_elements)
        else:
            sparsity_achieved = 0.0
            computation_saving = 0.0
        
        # Update layer statistics
        layer_stats = self.gradient_stats['layer_stats'][layer_name]
        layer_stats['total_gradients'] += total_elements
        layer_stats['pruned_gradients'] += (original_nonzero - sparse_nonzero)
        layer_stats['average_magnitude'] = (
            layer_stats['average_magnitude'] + original_grad.abs().mean().item()
        ) / 2
        
        # Update global statistics
        self.gradient_stats['average_sparsity_achieved'] = (
            self.gradient_stats['average_sparsity_achieved'] + sparsity_achieved
        ) / 2
        
        self.gradient_stats['gradient_computation_savings'] = (
            self.gradient_stats['gradient_computation_savings'] + computation_saving
        ) / 2
    
    def dynamic_threshold_adjustment(self, performance_metric: Callable,
                                  target_performance: float):
        """Dynamically adjust pruning thresholds based on performance"""
        
        def adjustment_callback(current_performance):
            if current_performance < target_performance * 0.9:
                # Performance dropping, reduce sparsity
                new_sparsity = max(self.config.sparsity_level - 0.1, 0.0)
                if new_sparsity != self.config.sparsity_level:
                    print(f"   Reducing backprop sparsity to {new_sparsity:.1%}")
                    self.config.sparsity_level = new_sparsity
            elif current_performance > target_performance * 1.1:
                # Performance good, can increase sparsity
                new_sparsity = min(self.config.sparsity_level + 0.05, 0.9)
                if new_sparsity != self.config.sparsity_level:
                    print(f"   Increasing backprop sparsity to {new_sparsity:.1%}")
                    self.config.sparsity_level = new_sparsity
        
        return adjustment_callback
    
    def get_backprop_stats(self) -> Dict[str, Any]:
        """Get sparse backpropagation statistics"""
        
        stats = self.gradient_stats.copy()
        
        # Calculate overall statistics
        total_gradients = 0
        total_pruned = 0
        
        for layer_stats in stats['layer_stats'].values():
            total_gradients += layer_stats['total_gradients']
            total_pruned += layer_stats['pruned_gradients']
        
        if total_gradients > 0:
            overall_sparsity = total_pruned / total_gradients
        else:
            overall_sparsity = 0.0
        
        stats.update({
            'overall_sparsity_achieved': overall_sparsity,
            'estimated_speedup': 1 / (1 - overall_sparsity) if overall_sparsity < 1 else 1.0,
            'memory_savings': overall_sparsity
        })
        
        return stats
    
    def disable_sparse_backprop(self):
        """Disable sparse backpropagation"""
        
        # Remove all backward hooks
        for module in self.model.modules():
            module._backward_hooks.clear()
        
        print("🔌 Sparse backpropagation disabled")
    
    def layer_analysis_report(self) -> Dict[str, Any]:
        """Generate analysis report for each layer"""
        
        report = {
            'layer_analysis': {},
            'recommendations': []
        }
        
        for layer_name, stats in self.gradient_stats['layer_stats'].items():
            if stats['total_gradients'] > 0:
                sparsity = stats['pruned_gradients'] / stats['total_gradients']
                
                report['layer_analysis'][layer_name] = {
                    'sparsity_achieved': sparsity,
                    'average_gradient_magnitude': stats['average_magnitude'],
                    'efficiency_score': self._calculate_layer_efficiency(stats)
                }
                
                # Generate recommendations
                if sparsity > 0.8:
                    report['recommendations'].append(
                        f"Layer {layer_name}: High sparsity achieved, consider increasing target"
                    )
                elif sparsity < 0.2:
                    report['recommendations'].append(
                        f"Layer {layer_name}: Low sparsity, consider reducing target or changing method"
                    )
        
        return report
    
    def _calculate_layer_efficiency(self, stats: Dict[str, Any]) -> float:
        """Calculate efficiency score for a layer"""
        
        if stats['total_gradients'] == 0:
            return 0.0
        
        sparsity = stats['pruned_gradients'] / stats['total_gradients']
        magnitude = stats['average_magnitude']
        
        # Higher sparsity and reasonable magnitude = good efficiency
        efficiency = sparsity * min(magnitude * 100, 1.0)
        
        return efficiency
