# /src/training/dual_training_loop.py

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ..models.asreh_model import ASREHModel
from ..models.arlc_controller import ARLCController
from ..models.explainability_module import ExplainabilityModule
from ..data.tetris_generator import MemoryEfficientDataset, get_conceptual_features
from ..games.tetris_env import TetrisEnvironment, get_conceptual_features as get_env_conceptual_features
from ..utils.config import Config
import os

# Create a configuration object instance
config = Config()

def get_dataloader():
    """Initializes and returns a data loader for the Tetris domain."""
    print("Initializing Tetris dataset...")
    tetris_dataset = MemoryEfficientDataset(size=config.TETRIS_DATA_SIZE)
    tetris_loader = DataLoader(tetris_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() // 2 or 1)
    print(f"Dataset has {len(tetris_dataset)} samples.")
    return tetris_loader

def train_model():
    """
    Main training function that orchestrates the entire ASREH pipeline for Tetris.
    """
    print(f"Using device: {config.DEVICE}")
    
    # Initialize core components
    model = ASREHModel(in_channels=config.IN_CHANNELS, num_conceptual_features=4).to(config.DEVICE)
    arlc_controller = ARLCController(
        points_per_line=config.ARLC_REWARD_COEFFS['tetris']['lines_cleared'],
        gap_penalty=config.ARLC_REWARD_COEFFS['tetris']['gaps'],
        height_penalty=config.ARLC_REWARD_COEFFS['tetris']['max_height']
    )
    explainability_module = ExplainabilityModule(model)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Loss functions for each branch of the model
    # Supervised Loss for the World Model's prediction of the next board state
    world_model_loss_fn = nn.BCELoss() 
    # Reinforcement Loss for the ARLC's policy optimization
    policy_loss_fn = nn.MSELoss() 

    dataloader = get_dataloader()
    
    print("Starting Phase 1 training loop...")

    for epoch in range(config.NUM_EPOCHS):
        total_wm_loss = 0
        total_policy_loss = 0
        total_reward = 0
        
        for i, batch in enumerate(dataloader):
            state_before_img, board_after_img, next_piece_idx, conceptual_features_before = batch
            
            # Move data to the correct device
            state_before_img = state_before_img.to(config.DEVICE)
            board_after_img = board_after_img.to(config.DEVICE)
            conceptual_features_before = conceptual_features_before.to(config.DEVICE)

            # --- 1. World Model Training (Supervised Learning) ---
            # The model predicts the next board state
            predicted_board, fused_representation = model(state_before_img, conceptual_features_before, domain='tetris')
            wm_loss = world_model_loss_fn(predicted_board, board_after_img)
            
            # --- 2. ARLC Training (Reinforcement Learning) ---
            # Use the ARLC to simulate a move and get a reward signal
            # This is a simplified RL step for demonstration purposes
            # In a real scenario, this would be a more complex policy gradient or Q-learning update
            
            # The ARLC provides a "target" based on a heuristic
            # We'll calculate a simple reward based on the conceptual features of the predicted board
            conceptual_features_after_pred = get_env_conceptual_features(predicted_board.detach().cpu().squeeze().numpy())
            
            # Calculate a reward based on the heuristic
            reward = arlc_controller.evaluate_board_state(predicted_board.detach().cpu().squeeze().numpy())['score']
            
            # This is a simplified policy loss. In a real scenario, this would be more complex
            # We'll use MSE between the conceptual embedding and the reward signal
            reward_tensor = torch.tensor([reward], dtype=torch.float32).to(config.DEVICE)
            policy_loss = policy_loss_fn(fused_representation.mean(), reward_tensor)
            
            # --- 3. Combined Loss and Optimization ---
            total_loss = wm_loss + policy_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            total_wm_loss += wm_loss.item()
            total_policy_loss += policy_loss.item()
            total_reward += reward

            if (i + 1) % config.LOG_INTERVAL == 0:
                print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], Step [{i+1}/{len(dataloader)}], "
                      f"WM Loss: {wm_loss.item():.4f}, Policy Loss: {policy_loss.item():.4f}, "
                      f"Avg Reward: {total_reward/(i+1):.4f}")

        avg_wm_loss = total_wm_loss / len(dataloader)
        avg_policy_loss = total_policy_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Avg WM Loss: {avg_wm_loss:.4f}, Avg Policy Loss: {avg_policy_loss:.4f}")
        
    print("Training finished!")
    # Save the final model
    torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, "zenith_protocol_phase1.pth"))
    print(f"Model saved to {config.CHECKPOINT_DIR}/zenith_protocol_phase1.pth")


if __name__ == '__main__':
    train_model()

