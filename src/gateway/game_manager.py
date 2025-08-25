# /src/gateway/game_manager.py

import os
import torch
import sys
import time
import numpy as np
from ..models.asreh_model import ASREHModel
from ..models.arlc_controller import ARLCController
from ..models.explainability_module import ExplainabilityModule
from ..utils.config import Config
# Placeholder for game environments
from ..games.tetris_env import TetrisEnv
from ..games.chess_env import ChessEnv

def select_game():
    """Prompts the user to select a game and mode."""
    print("Welcome to the Zenith Protocol Game Gateway!")
    print("Please select a game:")
    print("1. Tetris")
    print("2. Chess")
    game_choice = input("Enter your choice (1 or 2): ")

    print("\nSelect a mode:")
    print("1. Play against the AI (VS Mode)")
    print("2. Watch the AI play (Spectator Mode)")
    mode_choice = input("Enter your choice (1 or 2): ")

    return game_choice, mode_choice

def load_game(game_choice: str):
    """Loads the appropriate game environment and model."""
    config = Config()
    if game_choice == '1':
        game_env = TetrisEnv()
        model_path = os.path.join(config.CHECKPOINT_DIR, "zenith_protocol_phase4_tetris.pth")
    elif game_choice == '2':
        game_env = ChessEnv()
        model_path = os.path.join(config.CHECKPOINT_DIR, "zenith_protocol_phase4_chess.pth")
    else:
        raise ValueError("Invalid game choice.")

    # Initialize model, ARLC, and EM
    model = ASREHModel().to(config.DEVICE)
    arlc = ARLCController()
    em = ExplainabilityModule(model)

    # Load pre-trained weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}. Starting with a fresh model.")

    return game_env, model, arlc, em

def start_game_loop(game_env, model, arlc, em, mode: str):
    """Starts the main game loop."""
    print(f"\nStarting game in {mode.capitalize()} Mode...")

    game_state = game_env.reset()
    last_fused_rep = None
    done = False
    
    while not done:
        # Get conceptual features for the current state
        conceptual_features = game_env.get_conceptual_features(game_state)
        
        # Run the model
        with torch.no_grad():
            state_tensor = torch.tensor(game_state).unsqueeze(0).float().to(Config.DEVICE)
            conceptual_tensor = torch.tensor(conceptual_features).unsqueeze(0).float().to(Config.DEVICE)
            
            # The model's forward pass
            predicted_output, fused_representation, _ = model(state_tensor, conceptual_tensor, game_env.domain)
        
        # Calculate EoM bonus
        eom_bonus = 0.0
        if last_fused_rep is not None:
            eom_bonus = arlc.calculate_eom_bonus(last_fused_rep, fused_representation)
            
        last_fused_rep = fused_representation.clone()
        
        # Get the AI's move
        chosen_move, decision_context = arlc.choose_move(game_state, game_env.domain)
        
        # Explain the decision
        decision_context['eom_bonus'] = eom_bonus
        explanation = em.generate_explanation(conceptual_tensor, fused_representation, decision_context, game_env.domain)
        
        print("\n--- AI's Turn ---")
        game_env.render(game_state)
        print("AI's Reasoning:", explanation['narrative'])

        # Update the game state based on the chosen move
        game_state, _, done, _ = game_env.step(chosen_move)

        if mode == 'vs':
            # Handle human's turn (this is a placeholder)
            print("Your turn...")
            human_move = input("Enter your move: ")
            game_state, _, done, _ = game_env.step(human_move) # Simplified step

        time.sleep(1) # Pause for a second for better viewing

def main():
    try:
        game_choice, mode_choice = select_game()
        game_env, model, arlc, em = load_game(game_choice)
        
        if mode_choice == '1':
            start_game_loop(game_env, model, arlc, em, mode='vs')
        elif mode_choice == '2':
            start_game_loop(game_env, model, arlc, em, mode='spectator')
        else:
            print("Invalid mode choice. Exiting.")
            sys.exit()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
  
