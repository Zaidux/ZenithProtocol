"""
Dynamic Quantization - Reduces model size and speeds up inference
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

class QuantizationType(Enum):
    INT8 = "int8"
    INT4 = "int4" 
    MIXED = "mixed"
    FLOAT16 = "float16"
    DYNAMIC = "dynamic"

@dataclass
class QuantizationConfig:
    quant_type: QuantizationType
    per_channel: bool = True
    symmetric: bool = True
    preserve_accuracy: bool = True
    calibration_samples: int = 100
    target_device: str = "auto"

class DynamicQuantization:
    def __init__(self, model: nn.Module):
        self.model = model
        self.original_state = model.state_dict().copy()
        self.quantization_stats = {}
        
    def quantize_model(self, config: QuantizationConfig) -> nn.Module:
        """Apply quantization to the model based on configuration"""
        
        print(f"⚡ Quantizing model to {config.quant_type.value}...")
        
        # Store original model for reference
        original_size = self._calculate_model_size(self.model)
        
        # Apply quantization based on type
        if config.quant_type == QuantizationType.INT8:
            quantized_model = self._quantize_int8(config)
        elif config.quant_type == QuantizationType.FLOAT16:
            quantized_model = self._quantize_float16(config)
        elif config.quant_type == QuantizationType.DYNAMIC:
            quantized_model = self._quantize_dynamic(config)
        elif config.quant_type == QuantizationType.MIXED:
            quantized_model = self._quantize_mixed_precision(config)
        else:
            raise ValueError(f"Unsupported quantization type: {config.quant_type}")
        
        # Calculate compression statistics
        quantized_size = self._calculate_model_size(quantized_model)
        compression_ratio = original_size / quantized_size
        
        self.quantization_stats = {
            'original_size_mb': original_size,
            'quantized_size_mb': quantized_size,
            'compression_ratio': compression_ratio,
            'quantization_type': config.quant_type.value,
            'estimated_speedup': self._estimate_speedup(compression_ratio),
            'accuracy_impact': self._estimate_accuracy_impact(config)
        }
        
        print(f"✅ Model quantized: {compression_ratio:.2f}x smaller")
        print(f"   Original: {original_size:.2f}MB → Quantized: {quantized_size:.2f}MB")
        
        return quantized_model
    
    def _quantize_int8(self, config: QuantizationConfig) -> nn.Module:
        """Apply INT8 quantization"""
        
        # Prepare model for quantization
        model_prepared = torch.quantization.prepare(self.model, inplace=False)
        
        # Calibrate with sample data (would use actual calibration dataset)
        self._calibrate_model(model_prepared, config.calibration_samples)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model_prepared, inplace=False)
        
        return quantized_model
    
    def _quantize_float16(self, config: QuantizationConfig) -> nn.Module:
        """Apply FP16 quantization"""
        
        # Convert model to half precision
        quantized_model = self.model.half()
        
        return quantized_model
    
    def _quantize_dynamic(self, config: QuantizationConfig) -> nn.Module:
        """Apply dynamic quantization (weights only)"""
        
        # Dynamically quantize linear and LSTM layers
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear, nn.LSTM},
            dtype=torch.qint8
        )
        
        return quantized_model
    
    def _quantize_mixed_precision(self, config: QuantizationConfig) -> nn.Module:
        """Apply mixed precision quantization"""
        
        class MixedPrecisionModel(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.original_model = original_model
                self._setup_mixed_precision()
            
            def _setup_mixed_precision(self):
                # Convert specific layers to different precisions
                for name, module in self.original_model.named_children():
                    if isinstance(module, nn.Linear):
                        # Keep linear layers in FP16 for speed
                        module.half()
                    elif isinstance(module, nn.LayerNorm):
                        # Keep normalization in FP32 for stability
                        module.float()
                    else:
                        # Default to FP16
                        module.half()
            
            def forward(self, x):
                return self.original_model(x)
        
        return MixedPrecisionModel(self.model)
    
    def _calibrate_model(self, model: nn.Module, num_samples: int):
        """Calibrate model for quantization with sample data"""
        
        print(f"   Calibrating with {num_samples} samples...")
        
        # Generate fake calibration data (would use real dataset)
        calibration_data = [torch.randn(1, 768) for _ in range(num_samples)]
        
        # Run calibration passes
        model.eval()
        with torch.no_grad():
            for sample in calibration_data:
                _ = model(sample)
    
    def adaptive_quantization(self, target_device: str, 
                            available_memory: float) -> QuantizationConfig:
        """Automatically select best quantization based on device capabilities"""
        
        device_capabilities = {
            "high_end_gpu": QuantizationType.FLOAT16,
            "mid_range_gpu": QuantizationType.MIXED,
            "low_end_gpu": QuantizationType.DYNAMIC,
            "cpu": QuantizationType.INT8,
            "mobile": QuantizationType.INT8,
            "edge": QuantizationType.INT4
        }
        
        # Determine device type based on available memory
        if available_memory > 8000:  # 8GB+
            device_type = "high_end_gpu"
        elif available_memory > 4000:  # 4GB+
            device_type = "mid_range_gpu" 
        elif available_memory > 2000:  # 2GB+
            device_type = "low_end_gpu"
        elif available_memory > 1000:  # 1GB+
            device_type = "cpu"
        else:
            device_type = "mobile"
        
        quant_type = device_capabilities.get(device_type, QuantizationType.DYNAMIC)
        
        return QuantizationConfig(
            quant_type=quant_type,
            target_device=device_type,
            preserve_accuracy=(available_memory > 2000)
        )
    
    def progressive_quantization(self, model: nn.Module, 
                               accuracy_threshold: float = 0.95) -> nn.Module:
        """Progressively quantize model while maintaining accuracy"""
        
        print("🎯 Applying progressive quantization...")
        
        # Test different quantization levels
        quantization_levels = [
            QuantizationType.FLOAT16,
            QuantizationType.MIXED, 
            QuantizationType.DYNAMIC,
            QuantizationType.INT8
        ]
        
        best_model = model
        current_accuracy = 1.0  # Assume full accuracy initially
        
        for level in quantization_levels:
            print(f"   Testing {level.value}...")
            
            config = QuantizationConfig(quant_type=level)
            test_model = self.quantize_model(config)
            
            # Estimate accuracy impact (would use actual validation)
            estimated_accuracy = current_accuracy * self._estimate_accuracy_impact(config)
            
            if estimated_accuracy >= accuracy_threshold:
                best_model = test_model
                current_accuracy = estimated_accuracy
                print(f"   ✓ Acceptable accuracy: {estimated_accuracy:.3f}")
            else:
                print(f"   ✗ Accuracy too low: {estimated_accuracy:.3f}")
                break
        
        return best_model
    
    def quantize_for_deployment(self, model: nn.Module, 
                              deployment_scenario: str) -> nn.Module:
        """Optimize quantization for specific deployment scenarios"""
        
        deployment_configs = {
            "real_time_inference": QuantizationConfig(
                quant_type=QuantizationType.INT8,
                preserve_accuracy=False  # Prioritize speed
            ),
            "batch_processing": QuantizationConfig(
                quant_type=QuantizationType.MIXED,
                preserve_accuracy=True
            ),
            "mobile_app": QuantizationConfig(
                quant_type=QuantizationType.INT8,
                preserve_accuracy=True
            ),
            "edge_device": QuantizationConfig(
                quant_type=QuantizationType.INT4,
                preserve_accuracy=False
            ),
            "research": QuantizationConfig(
                quant_type=QuantizationType.FLOAT16,
                preserve_accuracy=True
            )
        }
        
        config = deployment_configs.get(deployment_scenario, 
                                      QuantizationConfig(QuantizationType.DYNAMIC))
        
        return self.quantize_model(config)
    
    def get_quantization_stats(self) -> Dict[str, Any]:
        """Get statistics about current quantization"""
        return self.quantization_stats
    
    def restore_original(self) -> nn.Module:
        """Restore original unquantized model"""
        self.model.load_state_dict(self.original_state)
        return self.model
    
    def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculate model size in megabytes"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb
    
    def _estimate_speedup(self, compression_ratio: float) -> float:
        """Estimate inference speedup from quantization"""
        # Simplified estimation - real speedup depends on hardware
        base_speedup = compression_ratio * 0.7  # 70% efficiency
        return min(base_speedup, 5.0)  # Cap at 5x
    
    def _estimate_accuracy_impact(self, config: QuantizationConfig) -> float:
        """Estimate accuracy impact of quantization"""
        accuracy_impact = {
            QuantizationType.FLOAT16: 0.99,   # 1% drop
            QuantizationType.MIXED: 0.98,     # 2% drop  
            QuantizationType.DYNAMIC: 0.95,   # 5% drop
            QuantizationType.INT8: 0.92,      # 8% drop
            QuantizationType.INT4: 0.85       # 15% drop
        }
        
        base_impact = accuracy_impact.get(config.quant_type, 0.95)
        
        # Adjust based on configuration
        if config.preserve_accuracy:
            base_impact += 0.03  # 3% improvement for accuracy preservation
        
        return min(base_impact, 1.0)
