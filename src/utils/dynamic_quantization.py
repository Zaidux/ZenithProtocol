# /src/utils/dynamic_quantization.py

import torch
import torch.quantization
from typing import Dict, List, Tuple
from .config import Config
import time
from copy import deepcopy
import os

class DynamicQuantization:
    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        self.original_model_state_dict = deepcopy(model.state_dict())
        self.quantized_model = None
        self.is_quantized = False
        self.device = device

    def should_quantize(self, epoch: int, avg_loss: float, inference_time: float) -> bool:
        config = Config()

        if epoch >= config.QUANTIZATION_EPOCH_THRESHOLD:
            if avg_loss < config.QUANTIZATION_LOSS_THRESHOLD:
                print(f"Heuristic triggered: Loss ({avg_loss:.4f}) is low enough for quantization.")

                if inference_time > config.INFERENCE_LATENCY_THRESHOLD:
                    print(f"Heuristic triggered: Inference latency ({inference_time:.4f}s) is above threshold.")
                    return True
        return False

    def quantize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        if self.is_quantized:
            print("Model is already quantized. Skipping.")
            return self.quantized_model

        try:
            model.eval()
            torch.backends.quantized.engine = self.device
            fused_model = torch.quantization.fuse_modules(model,
                                                        [['shared_encoder.0', 'shared_encoder.1'],
                                                         ['shared_encoder.3', 'shared_encoder.4']],
                                                        inplace=True)
            prepared_model = torch.quantization.prepare_dynamic(fused_model, inplace=True)
            self.quantized_model = torch.quantization.convert(prepared_model, inplace=True)
            self.is_quantized = True

            original_size = self.get_model_size(model)
            quantized_size = self.get_model_size(self.quantized_model)
            print(f"Model quantized successfully. Original size: {original_size:.2f} MB, Quantized size: {quantized_size:.2f} MB")

            return self.quantized_model

        except Exception as e:
            print(f"Error during dynamic quantization: {e}")
            return model

    def unquantize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        if not self.is_quantized:
            print("Model is not quantized. Skipping.")
            return model

        print("Reverting model to original non-quantized state.")
        unquantized_model = model.__class__(*model._init_args)
        unquantized_model.load_state_dict(self.original_model_state_dict)
        self.is_quantized = False
        return unquantized_model

    @staticmethod
    def get_model_size(model: torch.nn.Module) -> float:
        torch.save(model.state_dict(), "temp_model_size.p")
        size_mb = os.path.getsize("temp_model_size.p") / 1e6
        os.remove("temp_model_size.p")
        return size_mb
