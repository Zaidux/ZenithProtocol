# /src/training/dual_training_loop.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..model.asreh_model import ASREHModel
from ..model.arlc_controller import ARLCController
from ..data.tetris_generator import MemoryEfficientDataset
from ..data.chess_generator import ChessDataset
from ..utils.config import Config

# Create a configuration object instance
config = Config()

def get_dataloaders():
    """Initializes and returns data loaders for both domains."""
    print("Initializing Tetris and Chess datasets...")
    tetris_dataset = MemoryEfficientDataset(size=config.TETRIS_DATA_SIZE)
    chess_dataset = ChessDataset(size=config.CHESS_DATA_SIZE)

    tetris_loader = DataLoader(tetris_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    chess_loader = DataLoader(chess_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    
    return tetris_loader, chess_loader

def train_model():
    """
    Main training function that runs the dual-domain training loop.
    """
    print(f"Using device: {config.DEVICE}")
    model = ASREHModel(hct_dim=config.HCT_DIM).to(config.DEVICE)
    arlc_controller = ARLCController(model, lr=config.LEARNING_RATE)
    
    tetris_loader, chess_loader = get_dataloaders()
    
    # We will alternate between training on Tetris and Chess data
    tetris_iter = iter(tetris_loader)
    chess_iter = iter(chess_loader)

    print("Starting dual-domain training loop...")

    for epoch in range(config.NUM_EPOCHS):
        # --- Training on Tetris Domain ---
        try:
            tetris_batch = next(tetris_iter)
        except StopIteration:
            tetris_iter = iter(tetris_loader)
            tetris_batch = next(tetris_iter)
        
        # Prepare data for model
        state_before, board_after, next_piece_idx, conceptual_features_before = tetris_batch
        state_before = state_before.to(config.DEVICE)
        conceptual_features_before = conceptual_features_before.to(config.DEVICE)
        board_after = board_after.to(config.DEVICE)
        
        # Pass data to ARLC controller for a training step
        arlc_controller.train_step(
            (state_before, board_after, next_piece_idx, conceptual_features_before),
            domain='tetris'
        )
        
        # --- Training on Chess Domain ---
        try:
            chess_batch = next(chess_iter)
        except StopIteration:
            chess_iter = iter(chess_loader)
            chess_batch = next(chess_iter)

        # Prepare data for model
        state_before, move_idx, conceptual_features_before = chess_batch
        state_before = state_before.to(config.DEVICE)
        move_idx = move_idx.to(config.DEVICE)
        conceptual_features_before = conceptual_features_before.to(config.DEVICE)
        
        # Pass data to ARLC controller for a training step
        arlc_controller.train_step(
            (state_before, move_idx, conceptual_features_before),
            domain='chess'
        )

        if (epoch + 1) % config.LOG_INTERVAL == 0:
            print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] completed.")
            # In a real scenario, we would add more detailed logging here
            # e.g., a simple test to check performance on a validation set

    print("Training finished!")
    # Save the final model
    torch.save(model.state_dict(), f"{config.CHECKPOINT_DIR}/asreh_final.pth")
    print(f"Model saved to {config.CHECKPOINT_DIR}/asreh_final.pth")


if __name__ == '__main__':
    train_model()


