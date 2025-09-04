# src/image_generation/image_generator.py

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple

# Import the Python and C++ components
from ..conceptual_encoder.conceptual_encoder import ZenithConceptualEncoder
from ..conceptual_encoder.conceptual_visual_encoder import ZenithConceptualVisualEncoder
from ..models.explainability_module import ExplainabilityModule
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph

# Import the C++ backend for the image generation core
import image_generator_core_cpp
import asreh_model_cpp

class ZenithImageGenerator:
    """
    A text-to-image generator built on the Zenith Protocol's principles.
    It uses a conceptual understanding of a prompt to generate images that
    are consistent, causally-aware, and explainable.
    """
    def __init__(self,
                 conceptual_dim: int = 512,
                 image_width: int = 512,
                 image_height: int = 512,
                 ckg: ConceptualKnowledgeGraph = None):
        
        # Core Components
        self.ckg = ckg or ConceptualKnowledgeGraph()
        self.text_encoder = ZenithConceptualEncoder(embedding_dim=conceptual_dim, conceptual_map={})
        self.visual_encoder = ZenithConceptualVisualEncoder(embedding_dim=conceptual_dim)
        
        # The C++ core handles the high-performance generation loop
        self.cpp_image_core = image_generator_core_cpp.ImageGeneratorCore(
            conceptual_dim,
            image_width,
            image_height
        )
        
        # The Explainability Module is used to justify the generation process
        # We need a mock model and sswm for this example.
        self.explainability_module = ExplainabilityModule(
            model=None, # Placeholder
            sswm=None, # Placeholder
            ckg=self.ckg
        )
        
    def generate_image_from_prompt(self, prompt: str) -> Image.Image:
        """
        Generates an image from a text prompt.
        
        The process is:
        1. Encode the text prompt into a conceptual vector.
        2. Use the C++ backend to generate an initial image.
        3. Use the visual encoder to check for conceptual consistency.
        4. (Optional) Use the EM to generate a narrative.
        """
        print(f"[Generator] Encoding prompt: '{prompt}'")
        
        # Step 1: Encode the text prompt into a conceptual vector using the Python/C++ hybrid encoder.
        conceptual_vector_tensor = self.text_encoder(prompt)
        conceptual_vector_np = conceptual_vector_tensor.detach().cpu().numpy().squeeze(0)

        # Step 2: Use the C++ core to generate the initial image
        print("[Generator] Generating initial image from conceptual vector...")
        raw_image_data_np = self.cpp_image_core.generate_image(conceptual_vector_np)
        
        initial_image = Image.fromarray(raw_image_data_np)
        
        # Step 3: Ensure conceptual consistency (e.g., check for persistent colors)
        print("[Generator] Verifying image for conceptual consistency...")
        consistent_image_data_np = self.cpp_image_core.ensure_consistency(raw_image_data_np, conceptual_vector_np)
        
        final_image = Image.fromarray(consistent_image_data_np)
        
        # Step 4: Generate an explanation for the image
        explanation_narrative = self._generate_image_explanation(prompt, conceptual_vector_np)
        print(f"\n[Generator] Explanation: {explanation_narrative}")
        
        return final_image
        
    def _generate_image_explanation(self, prompt: str, conceptual_vector: np.ndarray) -> str:
        """
        Generates an explanation for the image, detailing the conceptual reasoning.
        """
        # This is a conceptual implementation of an explanation
        # It would analyze the top conceptual features that drove the generation.
        # For example, it would identify the most influential concepts from the prompt.
        
        # For now, we return a simple narrative
        return self.explainability_module.generate_explanation(
            conceptual_features=torch.from_numpy(conceptual_vector),
            fused_representation=torch.from_numpy(conceptual_vector), # Fused representation is the same here
            decision_context={}, # Placeholder context
            domain="image_generation"
        )['narrative']

if __name__ == '__main__':
    # Initialize the image generator
    image_generator = ZenithImageGenerator()
    
    # Generate an image from a sample prompt
    prompt_text = "A robot is writing a book in the desert."
    generated_image = image_generator.generate_image_from_prompt(prompt_text)
    
    # Save the generated image
    output_path = "generated_image.png"
    generated_image.save(output_path)
    print(f"\nImage saved to {output_path}")


