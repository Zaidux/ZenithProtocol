# /src/training/dual_training_loop.py

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from ..models.asreh_model import ASREHModel
from ..models.arlc_controller import ARLCController
from ..models.explainability_module import ExplainabilityModule
from ..data.tetris_generator import MemoryEfficientDataset, get_conceptual_features as tetris_features
from ..data.chess_generator import ChessDataset, get_conceptual_features as chess_features
from ..utils.config import Config
import os
import itertools
import numpy as np

# Create a configuration object instance
config = Config()

def get_dataloaders():
    """Initializes and returns data loaders for both domains."""
    print("Initializing Tetris and Chess datasets...")
    tetris_dataset = MemoryEfficientDataset(size=config.TETRIS_DATA_SIZE)
    chess_dataset = ChessDataset(size=config.CHESS_DATA_SIZE)
    
    tetris_loader = DataLoader(tetris_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() // 2 or 1)
    chess_loader = DataLoader(chess_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() // 2 or 1)
    
    return tetris_loader, chess_loader

def train_model():
    """
    Main training function that orchestrates the entire ASREH pipeline for both domains.
    """
    print(f"Using device: {config.DEVICE}")
    
    # Initialize core components
    model = ASREHModel(
        in_channels=config.IN_CHANNELS, 
        num_tetris_features=4, 
        num_chess_features=5
    ).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Loss functions for each branch of the model
    tetris_wm_loss_fn = nn.BCELoss() 
    chess_policy_loss_fn = nn.CrossEntropyLoss()
    
    tetris_loader, chess_loader = get_dataloaders()
    
    # We will use itertools.cycle to alternate between the two data loaders
    tetris_iter = itertools.cycle(tetris_loader)
    chess_iter = itertools.cycle(chess_loader)
    
    print("Starting Phase 2: Split Mind training loop...")

    for epoch in range(config.NUM_EPOCHS):
        total_tetris_loss = 0
        total_chess_loss = 0

        num_batches = len(tetris_loader) + len(chess_loader)
        
        for i in range(num_batches):
            # --- 1. Train on Tetris Domain ---
            if i % 2 == 0:
                try:
                    state_before_img, board_after_img, _, conceptual_features = next(tetris_iter)
                    domain = 'tetris'
                except StopIteration:
                    tetris_iter = itertools.cycle(tetris_loader)
                    state_before_img, board_after_img, _, conceptual_features = next(tetris_iter)
                    domain = 'tetris'

                # Move data to the correct device
                state_before_img = state_before_img.to(config.DEVICE)
                board_after_img = board_after_img.to(config.DEVICE)
                conceptual_features = conceptual_features.to(config.DEVICE)

                # Predict and calculate loss
                predicted_board, fused_representation = model(state_before_img, conceptual_features, domain)
                loss = tetris_wm_loss_fn(predicted_board, board_after_img)
                total_tetris_loss += loss.item()

            # --- 2. Train on Chess Domain ---
            else:
                try:
                    state_before_img, move_idx, conceptual_features = next(chess_iter)
                    domain = 'chess'
                except StopIteration:
                    chess_iter = itertools.cycle(chess_loader)
                    state_before_img, move_idx, conceptual_features = next(chess_iter)
                    domain = 'chess'
                
                # Move data to the correct device
                state_before_img = state_before_img.to(config.DEVICE)
                move_idx = move_idx.to(config.DEVICE)
                conceptual_features = conceptual_features.to(config.DEVICE)

                # Predict and calculate loss
                predicted_move_logits, fused_representation = model(state_before_img, conceptual_features, domain)
                loss = chess_policy_loss_fn(predicted_move_logits, move_idx)
                total_chess_loss += loss.item()
            
            # --- 3. Combined Optimization ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % config.LOG_INTERVAL == 0:
                print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], Step [{i+1}/{num_batches}], "
                      f"Domain: {domain.capitalize()}, Loss: {loss.item():.4f}")

        avg_tetris_loss = total_tetris_loss / len(tetris_loader)
        avg_chess_loss = total_chess_loss / len(chess_loader)
        print(f"Epoch {epoch+1} finished. Avg Tetris Loss: {avg_tetris_loss:.4f}, Avg Chess Loss: {avg_chess_loss:.4f}")
        
    print("Training finished!")
    torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, "zenith_protocol_phase2.pth"))
    print(f"Model saved to {config.CHECKPOINT_DIR}/zenith_protocol_phase2.pth")

if __name__ == '__main__':
    train_model()
