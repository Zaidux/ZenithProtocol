# /src/utils/config.py

import torch

class Config:
    """
    Central configuration class for the entire ASREH project.
    """
    
    # --- General Project Settings ---
    PROJECT_NAME = "Adaptive Self-Regulating Explainable Hybrid (ASREH) AI"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Data Generation Settings ---
    DATA_DIR = "src/data/"
    CHESS_DATA_SIZE = 500000
    TETRIS_DATA_SIZE = 500000

    # --- Model Hyper-parameters ---
    HCT_DIM = 64
    
    # --- Training Hyper-parameters ---
    NUM_EPOCHS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    
    # --- ARLC Hyper-parameters ---
    # Coefficients for the conceptual reward function
    ARLC_REWARD_COEFFS = {
        'chess': {
            'material_advantage': 1.5,
            'king_safety': -2.0,
            'center_control': 1.0,
        },
        'tetris': {
            'lines_cleared': 1.0,
            'gaps': -0.5,
            'max_height': -0.8
        }
    }
    
    # --- Checkpoint and Logging ---
    CHECKPOINT_DIR = "checkpoints/"
    LOG_DIR = "logs/"
    LOG_INTERVAL = 100 # Log every 100 steps
    
    # --- Environment Settings ---
    CHESS_IMAGE_SIZE = (8, 8)  # Chess is a tensor, not a true image
    TETRIS_IMAGE_SIZE = (20, 10) # Tetris is a visual board
    
    def __init__(self):
        """Initializes configuration settings."""
        self.validate_settings()
        
    def validate_settings(self):
        """
        Validates key configuration settings to ensure they are consistent.
        """
        if self.BATCH_SIZE < 1:
            raise ValueError("BATCH_SIZE must be at least 1.")
        
        if self.LEARNING_RATE <= 0:
            raise ValueError("LEARNING_RATE must be a positive value.")
        
        # Add more validation checks as the project grows
        
# Create a singleton instance of the Config class
# This ensures all modules in the project use the same settings.
config = Config()

