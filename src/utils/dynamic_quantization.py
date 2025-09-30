# /src/utils/dynamic_quantization.py

import torch
import torch.quantization
from typing import Dict, List, Tuple
from .config import Config
import time
from copy import deepcopy
import os
import numpy as np

class DynamicQuantization:
    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        self.original_model_state_dict = deepcopy(model.state_dict())
        self.quantized_model = None
        self.is_quantized = False
        self.device = device
        self.config = Config()
        
        # Sparse attention aware quantization tracking
        self.sparse_attention_preserved = False
        self.quantization_metrics = {
            'original_size': 0,
            'quantized_size': 0,
            'inference_speedup': 0,
            'sparsity_preservation': 0
        }

    def should_quantize(self, epoch: int, avg_loss: float, inference_time: float, sparse_efficiency: float = 0.0) -> bool:
        """Enhanced quantization decision with sparse attention awareness"""
        
        # Base quantization criteria
        base_quantize = (
            epoch >= self.config.QUANTIZATION_EPOCH_THRESHOLD and
            avg_loss < self.config.QUANTIZATION_LOSS_THRESHOLD and
            inference_time > self.config.INFERENCE_LATENCY_THRESHOLD
        )
        
        # Sparse attention enhanced criteria
        if self.config.ZENITH_SPARSE_QUANTIZATION_ENABLED:
            sparse_quantize = (
                sparse_efficiency > self.config.ZENITH_QUANTIZATION_EFFICIENCY_THRESHOLD and
                epoch > self.config.ZENITH_QUANTIZATION_MIN_EPOCH
            )
            return base_quantize or sparse_quantize
        
        return base_quantize

    def quantize_model(self, model: torch.nn.Module, preserve_sparse_attention: bool = True) -> torch.nn.Module:
        """Enhanced quantization that preserves sparse attention patterns"""
        if self.is_quantized:
            print("Model is already quantized. Skipping.")
            return self.quantized_model

        try:
            print("Starting dynamic quantization with sparse attention preservation...")
            model.eval()
            
            # Store original model size
            self.quantization_metrics['original_size'] = self.get_model_size(model)
            
            # Preserve sparse attention patterns if present
            sparse_patterns = None
            if preserve_sparse_attention and hasattr(model, 'zenith_attention'):
                print("Preserving sparse attention patterns during quantization...")
                sparse_patterns = self._extract_sparse_patterns(model.zenith_attention)
                self.sparse_attention_preserved = True

            # Fuse modules for better quantization
            fused_model = self._fuse_model_modules(model)
            
            # Prepare for dynamic quantization
            if hasattr(torch.quantization, 'prepare_dynamic'):
                prepared_model = torch.quantization.prepare_dynamic(
                    fused_model, 
                    {torch.nn.Linear, torch.nn.Conv2d},
                    inplace=False
                )
            else:
                prepared_model = fused_model

            # Convert to quantized model
            self.quantized_model = torch.quantization.convert(prepared_model, inplace=False)
            
            # Restore sparse attention patterns
            if sparse_patterns and hasattr(self.quantized_model, 'zenith_attention'):
                self._restore_sparse_patterns(self.quantized_model.zenith_attention, sparse_patterns)
                print("Sparse attention patterns restored in quantized model.")

            self.is_quantized = True

            # Calculate quantization metrics
            self.quantization_metrics['quantized_size'] = self.get_model_size(self.quantized_model)
            size_reduction = 1 - (self.quantization_metrics['quantized_size'] / self.quantization_metrics['original_size'])
            
            # Measure inference speedup
            speedup = self._measure_inference_speedup(model, self.quantized_model)
            self.quantization_metrics['inference_speedup'] = speedup
            
            # Calculate sparsity preservation
            if self.sparse_attention_preserved:
                original_sparsity = self._measure_model_sparsity(model)
                quantized_sparsity = self._measure_model_sparsity(self.quantized_model)
                self.quantization_metrics['sparsity_preservation'] = quantized_sparsity / original_sparsity if original_sparsity > 0 else 1.0

            print(f"Dynamic quantization completed successfully!")
            print(f"Size reduction: {size_reduction:.2%} "
                  f"({self.quantization_metrics['original_size']:.2f}MB -> {self.quantization_metrics['quantized_size']:.2f}MB)")
            print(f"Inference speedup: {speedup:.2f}x")
            if self.sparse_attention_preserved:
                print(f"Sparsity preservation: {self.quantization_metrics['sparsity_preservation']:.2%}")

            return self.quantized_model

        except Exception as e:
            print(f"Error during dynamic quantization: {e}")
            print("Falling back to original model...")
            return model

    def _extract_sparse_patterns(self, zenith_attention):
        """Extract sparse attention patterns for preservation during quantization"""
        patterns = {}
        if hasattr(zenith_attention, 'get_sparsity_stats'):
            patterns['sparsity_stats'] = zenith_attention.get_sparsity_stats()
        
        if hasattr(zenith_attention, 'get_current_sparsity_pattern'):
            patterns['current_pattern'] = zenith_attention.get_current_sparsity_pattern()
        
        # Extract relevant parameters
        trainable_params = {}
        for name, param in zenith_attention.named_parameters():
            if param.requires_grad:
                trainable_params[name] = param.data.clone()
        
        patterns['trainable_params'] = trainable_params
        return patterns

    def _restore_sparse_patterns(self, zenith_attention, patterns):
        """Restore sparse attention patterns after quantization"""
        try:
            # Restore trainable parameters
            if 'trainable_params' in patterns:
                for name, param_data in patterns['trainable_params'].items():
                    if hasattr(zenith_attention, name):
                        param = getattr(zenith_attention, name)
                        if param.requires_grad:
                            param.data.copy_(param_data)
            
            # Re-initialize sparsity tracking
            if hasattr(zenith_attention, 'reset_sparsity_tracking'):
                zenith_attention.reset_sparsity_tracking()
                
            print("Sparse attention patterns successfully restored.")
        except Exception as e:
            print(f"Warning: Could not fully restore sparse patterns: {e}")

    def _fuse_model_modules(self, model):
        """Fuse model modules for better quantization efficiency"""
        try:
            # Common fusion patterns for transformer-like architectures
            fusion_candidates = []
            
            # Check for common module patterns
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Sequential):
                    if len(module) >= 2:
                        # Check for Linear + ReLU patterns
                        if (isinstance(module[0], torch.nn.Linear) and isinstance(module[1], torch.nn.ReLU)):
                            fusion_candidates.append(name)
            
            # Apply fusion
            if fusion_candidates:
                torch.quantization.fuse_modules(model, fusion_candidates, inplace=True)
                print(f"Fused {len(fusion_candidates)} module groups for quantization")
            
            return model
        except Exception as e:
            print(f"Module fusion failed: {e}. Continuing without fusion.")
            return model

    def _measure_inference_speedup(self, original_model, quantized_model, num_runs: int = 100):
        """Measure the inference speedup from quantization"""
        try:
            # Create dummy input
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            
            # Warm up
            with torch.no_grad():
                _ = original_model(dummy_input)
                _ = quantized_model(dummy_input)
            
            # Measure original model time
            start_time = time.time()
            for _ in range(num_runs):
                with torch.no_grad():
                    _ = original_model(dummy_input)
            original_time = time.time() - start_time
            
            # Measure quantized model time
            start_time = time.time()
            for _ in range(num_runs):
                with torch.no_grad():
                    _ = quantized_model(dummy_input)
            quantized_time = time.time() - start_time
            
            return original_time / quantized_time if quantized_time > 0 else 1.0
            
        except Exception as e:
            print(f"Speed measurement failed: {e}")
            return 1.0

    def _measure_model_sparsity(self, model):
        """Measure the sparsity of model parameters"""
        total_params = 0
        zero_params = 0
        
        for param in model.parameters():
            if param.dim() > 1:  # Only weight matrices, not biases
                total_params += param.numel()
                zero_params += (param == 0).sum().item()
        
        return zero_params / total_params if total_params > 0 else 0

    def unquantize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Revert to original non-quantized model"""
        if not self.is_quantized:
            print("Model is not quantized. Skipping.")
            return model

        print("Reverting model to original non-quantized state.")
        try:
            # Create new instance and load original state
            unquantized_model = model.__class__(*model._init_args)
            unquantized_model.load_state_dict(self.original_model_state_dict)
            self.is_quantized = False
            self.sparse_attention_preserved = False
            
            print("Model successfully unquantized.")
            return unquantized_model
            
        except Exception as e:
            print(f"Error during unquantization: {e}")
            return model

    def get_quantization_report(self) -> Dict:
        """Generate a comprehensive quantization report"""
        return {
            'is_quantized': self.is_quantized,
            'sparse_attention_preserved': self.sparse_attention_preserved,
            'metrics': self.quantization_metrics,
            'size_reduction': 1 - (self.quantization_metrics['quantized_size'] / self.quantization_metrics['original_size']),
            'recommendation': self._generate_quantization_recommendation()
        }

    def _generate_quantization_recommendation(self) -> str:
        """Generate recommendation based on quantization results"""
        if not self.is_quantized:
            return "Model is not quantized. Consider quantization for inference optimization."
        
        speedup = self.quantization_metrics['inference_speedup']
        size_reduction = 1 - (self.quantization_metrics['quantized_size'] / self.quantization_metrics['original_size'])
        
        if speedup > 2.0 and size_reduction > 0.5:
            return "Excellent quantization results. Highly recommended for deployment."
        elif speedup > 1.5 and size_reduction > 0.3:
            return "Good quantization results. Recommended for most use cases."
        elif speedup > 1.1:
            return "Moderate improvements. Consider context before deployment."
        else:
            return "Minimal benefits. May not be worth the precision loss."

    @staticmethod
    def get_model_size(model: torch.nn.Module) -> float:
        """Get model size in megabytes"""
        torch.save(model.state_dict(), "temp_model_size.p")
        size_mb = os.path.getsize("temp_model_size.p") / 1e6
        os.remove("temp_model_size.p")
        return size_mb

    @staticmethod
    def analyze_model_for_quantization(model: torch.nn.Module) -> Dict:
        """Analyze model suitability for quantization"""
        analysis = {
            'total_parameters': 0,
            'quantizable_layers': 0,
            'sparse_layers': 0,
            'suitability_score': 0.0
        }
        
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                analysis['quantizable_layers'] += 1
                analysis['total_parameters'] += sum(p.numel() for p in module.parameters())
                
                # Check for sparsity
                for param in module.parameters():
                    if param.dim() > 1:
                        sparsity = (param == 0).sum().item() / param.numel()
                        if sparsity > 0.1:  # 10% sparsity threshold
                            analysis['sparse_layers'] += 1
        
        # Calculate suitability score
        if analysis['quantizable_layers'] > 0:
            analysis['suitability_score'] = min(1.0, 
                (analysis['quantizable_layers'] / 10) *  # More layers = better
                (1 + analysis['sparse_layers'] / max(1, analysis['quantizable_layers']))  # Sparsity bonus
            )
        
        return analysis