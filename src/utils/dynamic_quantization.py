# /src/utils/dynamic_quantization.py

import torch
import torch.quantization
from typing import Dict, List, Tuple
from .config import Config

class DynamicQuantization:
    """
    Manages the dynamic quantization of the model.
    The model can decide when to quantize itself based on a heuristic,
    or a human can manually trigger it.
    """
    @staticmethod
    def should_quantize(epoch: int, avg_tetris_loss: float, avg_chess_loss: float) -> bool:
        """
        Determines whether the model should be dynamically quantized based on heuristics.
        Returns True if the model is deemed stable enough for quantization.
        """
        config = Config()

        # Heuristic 1: Quantize after a certain number of epochs to allow for stable training
        if epoch >= config.QUANTIZATION_EPOCH_THRESHOLD:
            # Heuristic 2: Quantize only if the loss is low and stable
            combined_loss = avg_tetris_loss + avg_chess_loss
            if combined_loss < 0.5: # A configurable threshold
                print(f"Heuristic triggered: Combined loss ({combined_loss:.4f}) is low enough for quantization.")
                return True

        return False

    @staticmethod
    def quantize_model(model: torch.nn.Module) -> torch.nn.Module:
        """
        Applies dynamic quantization to the specified model.
        This modifies the model in-place.
        """
        try:
            # Set the backend for quantization
            torch.backends.quantized.engine = 'qnnpack'
            
            # Fuse modules for better performance
            fused_model = torch.quantization.fuse_modules(model,
                                                        [['shared_encoder.0', 'shared_encoder.1'],
                                                         ['shared_encoder.3', 'shared_encoder.4']],
                                                        inplace=True)
            
            # Prepare the model for dynamic quantization
            prepared_model = torch.quantization.prepare_dynamic(fused_model, inplace=True)
            
            # Apply dynamic quantization
            quantized_model = torch.quantization.convert(prepared_model, inplace=True)
            
            return quantized_model
        
        except Exception as e:
            print(f"Error during dynamic quantization: {e}")
            return model
          
