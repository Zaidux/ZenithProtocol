# /src/utils/dynamic_quantization.py

import torch
import torch.quantization
from typing import Dict, List, Tuple
from .config import Config
import time
from copy import deepcopy

class DynamicQuantization:
    """
    Manages the dynamic quantization of the model.
    The model can decide when to quantize itself based on a heuristic,
    or a human can manually trigger it.
    """
    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        self.original_model_state_dict = deepcopy(model.state_dict())
        self.quantized_model = None
        self.is_quantized = False
        self.device = device
        
    def should_quantize(self, epoch: int, avg_loss: float, inference_time: float) -> bool:
        """
        Determines whether the model should be dynamically quantized based on heuristics.
        Returns True if the model is deemed stable enough for quantization.
        """
        config = Config()

        # Heuristic 1: Quantize after a certain number of epochs
        if epoch >= config.QUANTIZATION_EPOCH_THRESHOLD:
            # Heuristic 2: Quantize if the average loss is low and stable.
            if avg_loss < config.QUANTIZATION_LOSS_THRESHOLD:
                print(f"Heuristic triggered: Loss ({avg_loss:.4f}) is low enough for quantization.")
                
                # Heuristic 3: Quantize if inference time is too slow
                if inference_time > config.INFERENCE_LATENCY_THRESHOLD:
                    print(f"Heuristic triggered: Inference latency ({inference_time:.4f}s) is above threshold.")
                    return True
        return False

    def quantize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Applies dynamic quantization to the specified model and stores the quantized version.
        This modifies the model in-place.
        """
        if self.is_quantized:
            print("Model is already quantized. Skipping.")
            return self.quantized_model
        
        try:
            model.eval() # Set the model to evaluation mode for quantization
            # Set the backend for quantization
            torch.backends.quantized.engine = self.device
            
            # Fuse modules for better performance
            fused_model = torch.quantization.fuse_modules(model,
                                                        [['shared_encoder.0', 'shared_encoder.1'],
                                                         ['shared_encoder.3', 'shared_encoder.4']],
                                                        inplace=True)

            # Prepare the model for dynamic quantization
            prepared_model = torch.quantization.prepare_dynamic(fused_model, inplace=True)

            # Apply dynamic quantization
            self.quantized_model = torch.quantization.convert(prepared_model, inplace=True)
            self.is_quantized = True
            
            # Compare model sizes and provide a report
            original_size = self.get_model_size(model)
            quantized_size = self.get_model_size(self.quantized_model)
            print(f"Model quantized successfully. Original size: {original_size:.2f} MB, Quantized size: {quantized_size:.2f} MB")
            
            return self.quantized_model

        except Exception as e:
            print(f"Error during dynamic quantization: {e}")
            return model
            
    def unquantize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Reverts the model to its original, non-quantized state.
        """
        if not self.is_quantized:
            print("Model is not quantized. Skipping.")
            return model
            
        print("Reverting model to original non-quantized state.")
        unquantized_model = model.__class__(*model._init_args) # Assuming we can recreate the model
        unquantized_model.load_state_dict(self.original_model_state_dict)
        self.is_quantized = False
        return unquantized_model
        
    @staticmethod
    def get_model_size(model: torch.nn.Module) -> float:
        """Returns the size of the model in megabytes (MB)."""
        torch.save(model.state_dict(), "temp_model_size.p")
        size_mb = os.path.getsize("temp_model_size.p") / 1e6
        os.remove("temp_model_size.p")
        return size_mb


