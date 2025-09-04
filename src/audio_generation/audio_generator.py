# src/audio_generation/audio_generator.py

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional

# We'll use a library like SciPy for saving audio files from NumPy arrays.
# It is a common library for scientific computation in Python.
from scipy.io.wavfile import write as write_wav

# Import the Python and C++ components
from ..conceptual_encoder.conceptual_encoder import ZenithConceptualEncoder
from ..models.explainability_module import ExplainabilityModule
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph

# Import the C++ backend for the audio generation core
import audio_generator_core_cpp

class ZenithAudioGenerator:
    """
    A text-to-audio generator built on the Zenith Protocol's principles.
    It uses a conceptual understanding of a prompt to generate audio that
    is consistent, causally-aware, and explainable. It can also clone voices.
    """
    def __init__(self,
                 conceptual_dim: int = 512,
                 sample_rate: int = 44100,
                 buffer_size: int = 44100, # 1 second of audio
                 ckg: ConceptualKnowledgeGraph = None):
        
        # Core Components
        self.ckg = ckg or ConceptualKnowledgeGraph()
        self.text_encoder = ZenithConceptualEncoder(embedding_dim=conceptual_dim, conceptual_map={})

        # The C++ core handles the high-performance audio synthesis loop
        self.cpp_audio_core = audio_generator_core_cpp.AudioGeneratorCore(
            conceptual_dim,
            sample_rate,
            buffer_size
        )
        
        # The Explainability Module is used to justify the generation process
        # We need a mock model and sswm for this example.
        self.explainability_module = ExplainabilityModule(
            model=None, # Placeholder
            sswm=None, # Placeholder
            ckg=self.ckg
        )
        
        self.sample_rate = sample_rate

    def generate_audio_from_prompt(self, prompt: str, output_path: str = "generated_audio.wav") -> str:
        """
        Generates audio from a text prompt.
        
        The process is:
        1. Encode the text prompt into a conceptual vector.
        2. Use the C++ backend to generate an audio buffer.
        3. Save the audio buffer to a file.
        4. (Optional) Use the EM to generate a narrative.
        """
        print(f"[Generator] Encoding prompt: '{prompt}'")
        
        # Step 1: Encode the text prompt into a conceptual vector.
        conceptual_vector_tensor = self.text_encoder(prompt)
        conceptual_vector_np = conceptual_vector_tensor.detach().cpu().numpy().squeeze(0)

        # Step 2: Use the C++ core to generate the audio buffer
        print("[Generator] Generating audio from conceptual vector...")
        raw_audio_data_np = self.cpp_audio_core.generate_audio(conceptual_vector_np)
        
        # Step 3: Save the audio buffer to a WAV file
        print(f"[Generator] Saving audio to {output_path}...")
        write_wav(output_path, self.sample_rate, raw_audio_data_np)
        
        # Step 4: Generate an explanation for the audio
        explanation_narrative = self._generate_audio_explanation(prompt, conceptual_vector_np)
        print(f"\n[Generator] Explanation: {explanation_narrative}")
        
        return output_path

    def clone_voice_from_prompt(self, prompt: str, voice_sample_path: str, output_path: str = "cloned_voice.wav") -> str:
        """
        Clones a voice based on a text prompt and an audio sample.
        """
        # Load the voice sample using a library like PyAudio or SciPy
        # For this mock, we'll use a dummy array.
        try:
            voice_sample_np = np.random.randint(-32768, 32767, self.sample_rate, dtype=np.int16)
        except FileNotFoundError:
            print(f"Voice sample not found at {voice_sample_path}. Using mock data.")
            voice_sample_np = np.random.randint(-32768, 32767, self.sample_rate, dtype=np.int16)

        print(f"[Generator] Encoding prompt: '{prompt}' for voice cloning.")
        conceptual_vector_tensor = self.text_encoder(prompt)
        conceptual_vector_np = conceptual_vector_tensor.detach().cpu().numpy().squeeze(0)

        # Use the C++ core to clone the voice
        print("[Generator] Cloning voice...")
        cloned_audio_np = self.cpp_audio_core.clone_voice(conceptual_vector_np, voice_sample_np)
        
        # Save the cloned voice
        print(f"[Generator] Saving cloned voice to {output_path}...")
        write_wav(output_path, self.sample_rate, cloned_audio_np)
        
        # Generate an explanation
        explanation_narrative = self._generate_audio_explanation(f"voice cloning for prompt: '{prompt}'", conceptual_vector_np)
        print(f"\n[Generator] Explanation: {explanation_narrative}")
        
        return output_path

    def _generate_audio_explanation(self, prompt: str, conceptual_vector: np.ndarray) -> str:
        """
        Generates an explanation for the audio, detailing the conceptual reasoning.
        """
        return self.explainability_module.generate_explanation(
            conceptual_features=torch.from_numpy(conceptual_vector),
            fused_representation=torch.from_numpy(conceptual_vector),
            decision_context={},
            domain="audio_generation"
        )['narrative']

if __name__ == '__main__':
    audio_generator = ZenithAudioGenerator()

    # Generate music from a prompt
    music_path = audio_generator.generate_audio_from_prompt("a calm and soothing flute melody")
    print(f"\nMusic saved to {music_path}")

    # Clone a voice from a prompt and a mock sample
    cloned_voice_path = audio_generator.clone_voice_from_prompt("Hello, world!", "mock_sample.wav")
    print(f"Cloned voice saved to {cloned_voice_path}")
    https://www.youtube.com/watch?v=k-a2Wd0c2eU
      
