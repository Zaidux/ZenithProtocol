# /src/utils/config.py

import torch
import os

class Config:
    """
    Central configuration class for the entire ASREH project.
    Provides a single source of truth for all hyperparameters and settings.
    """
    def __init__(self):
        # --- General Project Settings ---
        self.PROJECT_NAME = "Adaptive Self-Regulating Explainable Hybrid (ASREH) AI"
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # --- Data Generation Settings ---
        self.DATA_DIR = os.path.join(os.getcwd(), 'data')
        self.CHESS_DATA_SIZE = 500000
        self.TETRIS_DATA_SIZE = 500000

        # --- Model Hyper-parameters ---
        self.HCT_DIM = 64
        self.IN_CHANNELS = 1 # Grayscale images for Tetris
        self.NUM_EXPERTS = 4 # New parameter for Mixture of Experts
        self.NUM_HCT_FEATURES = 10 # Maximum number of dynamically discovered concepts

        # --- Training Hyper-parameters ---
        self.NUM_EPOCHS = 10
        self.BATCH_SIZE = 64
        self.LEARNING_RATE = 1e-4

        # --- Federated Learning Settings ---
        self.FL_ROUNDS = 5        # Number of communication rounds
        self.NUM_CLIENTS = 10     # Total number of simulated clients
        self.CLIENTS_PER_ROUND = 3 # Number of clients to train per round

        # --- ARLC Hyper-parameters ---
        self.ARLC_REWARD_COEFFS = {
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
        self.EOM_WEIGHT = 2.0 # Energy of Movement bonus weight

        # --- Environment Settings ---
        self.TETRIS_BOARD_WIDTH = 10
        self.TETRIS_BOARD_HEIGHT = 20
        self.CHESS_IMAGE_SIZE = (8, 8)
        self.TETRIS_IMAGE_SIZE = (20, 10)

        # --- Checkpoint and Logging ---
        self.CHECKPOINT_DIR = os.path.join(os.getcwd(), 'checkpoints')
        self.LOG_DIR = os.path.join(os.getcwd(), 'logs')
        self.LOG_INTERVAL = 100 # Log every 100 steps
        self.QUANTIZATION_EPOCH_THRESHOLD = 5 # Epoch after which quantization can occur
        self.QUANTIZATION_LOSS_THRESHOLD = 0.5 # New: Loss must be below this to quantize
        self.INFERENCE_LATENCY_THRESHOLD = 0.1 # New: Quantize if latency is above this (in seconds)

        # --- Initialization ---
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
        
        if self.QUANTIZATION_LOSS_THRESHOLD < 0:
            raise ValueError("QUANTIZATION_LOSS_THRESHOLD must be a non-negative value.")

    def create_directories(self):
        """Creates necessary directories for logging and checkpoints."""
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)