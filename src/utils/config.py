# /src/utils/config.py

import torch
import os

class Config:
    """
    Central configuration class for the entire ASREH project.
    Provides a single source of truth for all hyperparameters and settings.
    """

    # --- General Project Settings ---
    PROJECT_NAME = "Adaptive Self-Regulating Explainable Hybrid (ASREH) AI"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Data Generation Settings ---
    DATA_DIR = os.path.join(os.getcwd(), 'data')
    CHESS_DATA_SIZE = 500000
    TETRIS_DATA_SIZE = 500000

    # --- Model Hyper-parameters ---
    HCT_DIM = 64
    IN_CHANNELS = 1 # Grayscale images for Tetris
    NUM_EXPERTS = 4 # New parameter for Mixture of Experts
    NUM_HCT_FEATURES = 10 # Maximum number of dynamically discovered concepts

    # --- Training Hyper-parameters ---
    NUM_EPOCHS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4

    # --- Federated Learning Settings ---
    FL_ROUNDS = 5        # Number of communication rounds
    NUM_CLIENTS = 10     # Total number of simulated clients
    CLIENTS_PER_ROUND = 3 # Number of clients to train per round

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
            'gaps': -5.0,
            'max_height': -1.0
        }
    }
    EOM_WEIGHT = 2.0 # Energy of Movement bonus weight

    # --- Environment Settings ---
    TETRIS_BOARD_WIDTH = 10
    TETRIS_BOARD_HEIGHT = 20
    CHESS_IMAGE_SIZE = (8, 8)
    TETRIS_IMAGE_SIZE = (20, 10)

    # --- Checkpoint and Logging ---
    CHECKPOINT_DIR = os.path.join(os.getcwd(), 'checkpoints')
    LOG_DIR = os.path.join(os.getcwd(), 'logs')
    LOG_INTERVAL = 100 # Log every 100 steps
    QUANTIZATION_EPOCH_THRESHOLD = 5 # Epoch after which quantization can occur

    def __init__(self):
        """Initializes configuration settings and creates necessary directories."""
        self.validate_settings()
        self.create_directories()

    def validate_settings(self):
        """
        Validates key configuration settings to ensure they are consistent.
        """
        if self.BATCH_SIZE < 1:
            raise ValueError("BATCH_SIZE must be at least 1.")

        if self.LEARNING_RATE <= 0:
            raise ValueError("LEARNING_RATE must be a positive value.")

    def create_directories(self):
        """Creates necessary directories for logging and checkpoints."""
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)

# Create a singleton instance of the Config class
config = Config()
