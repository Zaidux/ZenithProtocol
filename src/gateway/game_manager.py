# /src/gateway/game_manager.py

import os
import torch
import sys
import time
import numpy as np
from ..models.asreh_model import ASREHModel
from ..models.arlc_controller import ARLCController
from ..models.explainability_module import ExplainabilityModule
from ..models.strategic_planner import StrategicPlanner
from ..models.sswm import SSWM
from ..models.adversarial_module import AdversarialModule # New Import
from ..training.federated_learning import FederatedLearner
from ..training.meta_learner import MetaLearner
from ..utils.config import Config
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph # New Import

# Placeholder for game environments
from ..games.tetris_env import TetrisEnvironment as TetrisEnv
from ..games.chess_env import ChessEnvironment as ChessEnv

def select_game():
    print("Welcome to the Zenith Protocol Game Gateway!")
    print("Please select a game:")
    print("1. Tetris")
    print("2. Chess")
    print("3. Autonomous Exploration (New Domain)")
    print("4. Start Federated Learning")
    print("5. Start Meta-Learning")
    print("6. Run Adversarial Attack") # New option for Phase 6
    game_choice = input("Enter your choice (1, 2, 3, 4, 5, or 6): ")

    if game_choice == '3':
        mode_choice = 'exploration'
    elif game_choice == '4':
        mode_choice = 'federated_learning'
    elif game_choice == '5':
        mode_choice = 'meta_learning'
    elif game_choice == '6':
        mode_choice = 'adversarial'
    else:
        print("\nSelect a mode:")
        print("1. Play against the AI (VS Mode)")
        print("2. Watch the AI play (Spectator Mode)")
        mode_choice = input("Enter your choice (1 or 2): ")

    return game_choice, mode_choice

def load_game(game_choice: str):
    config = Config()
    model_path = None
    ckg = ConceptualKnowledgeGraph() # Initialize the CKG once

    if game_choice == '1':
        game_env = TetrisEnv(ckg=ckg) # Pass CKG to env
        model_path = os.path.join(config.CHECKPOINT_DIR, "zenith_protocol_phase5_tetris.pth")
    elif game_choice == '2':
        game_env = ChessEnv(ckg=ckg) # Pass CKG to env
        model_path = os.path.join(config.CHECKPOINT_DIR, "zenith_protocol_phase5_chess.pth")
    elif game_choice in ['3', '6']:
        game_env = TetrisEnv(ckg=ckg) # Default for exploration/adversarial
    elif game_choice in ['4', '5']:
        game_env = None
    else:
        raise ValueError("Invalid game choice.")

    model = ASREHModel().to(config.DEVICE)
    strategic_planner = StrategicPlanner(model)
    sswm = SSWM(input_dim=model.hct_dim).to(config.DEVICE)
    arlc = ARLCController(strategic_planner, sswm, ckg) # Pass CKG to ARLC
    em = ExplainabilityModule(model, sswm, ckg) # Pass CKG to EM
    adversary = AdversarialModule(model) # Initialize the Adversarial Module

    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        print(f"Loaded model from {model_path}")
    elif game_choice not in ['3', '4', '5', '6']:
        print(f"Warning: Model not found at {model_path}. Starting with a fresh model.")

    return game_env, model, arlc, em, adversary

def start_game_loop(game_env, model, arlc, em, mode: str):
    print(f"\nStarting game in {mode.capitalize()} Mode...")
    game_state = game_env.reset()
    done = False
    
    while not done:
        # Get conceptual features
        conceptual_features = game_env.evaluate_conceptual_features()
        with torch.no_grad():
            state_tensor = torch.tensor(game_state).unsqueeze(0).float().to(Config.DEVICE)
            conceptual_tensor = torch.tensor(conceptual_features).unsqueeze(0).float().to(Config.DEVICE)
            predicted_output, fused_representation, moe_context = model(state_tensor, conceptual_tensor, game_env.__class__.__name__)

        # Get AI's move
        chosen_move, decision_context = arlc.choose_move(game_state, game_env.__class__.__name__)
        decision_context['moe_context'] = moe_context
        explanation = em.generate_explanation(conceptual_tensor, fused_representation, decision_context, game_env.__class__.__name__)
        
        print("\n--- AI's Turn ---")
        print("AI's Reasoning:", explanation['narrative'])

        # User interaction loop
        while True:
            user_query = input("Ask the AI about its reasoning (e.g., 'explain', 'what's the strategy?', 'what if I made move 5?'): ")
            if user_query.lower() in ['exit', 'quit']:
                break
            response = em.handle_query(
                user_query,
                decision_context,
                conceptual_tensor,
                game_env.__class__.__name__,
                fused_representation
            )
            print(f"AI's Response: {response}")

        # Update the game state
        game_state_before = game_state
        game_state_after, _, _, done, conceptual_impact = game_env.step(chosen_move)
        
        # ARLC receives reward and logs to CKG
        reward = arlc.evaluate_reward(conceptual_impact, game_state_before, game_state_after)
        arlc.log_to_ckg(game_state_before, game_state_after, conceptual_impact, reward)

        if mode == 'vs':
            # Simplified human turn
            print("Your turn...")
            human_move = input("Enter your move: ")
            game_state, _, _, done, _ = game_env.step(human_move)

        game_state = game_state_after
        time.sleep(1)

def start_exploration_loop(game_env, model, arlc, em):
    print("\nStarting Autonomous Exploration...")
    arlc.is_exploring = True
    new_domain = 'new_domain' # Placeholder for a new domain

    print("Generating a small dataset for rapid adaptation...")
    new_domain_data = [
        {'state': np.random.rand(1, 20, 10), 'conceptual_features': np.random.rand(64), 'target': np.random.rand(64 * 64), 'domain': new_domain}
        for _ in range(5)
    ]
    arlc.rapid_adaptation_to_new_domain(new_domain_data)

    game_state = game_env.reset()
    done = False
    
    while not done:
        conceptual_features = arlc.get_generic_conceptual_features(game_state.shape)
        with torch.no_grad():
            state_tensor = torch.tensor(game_state).unsqueeze(0).float().to(Config.DEVICE)
            conceptual_tensor = torch.tensor(conceptual_features).unsqueeze(0).float().to(Config.DEVICE)
            predicted_output, fused_representation, moe_context = model(state_tensor, conceptual_tensor, new_domain)

        chosen_move, decision_context = arlc.choose_move(game_state, new_domain)
        decision_context['moe_context'] = moe_context
        
        game_state_before = game_state
        game_state_after, _, _, done, conceptual_impact = game_env.step(chosen_move)

        reward = arlc.evaluate_reward(conceptual_impact, game_state_before, game_state_after)
        arlc.log_to_ckg(game_state_before, game_state_after, conceptual_impact, reward, domain=new_domain)

        explanation = em.generate_explanation(conceptual_tensor, fused_representation, decision_context, new_domain)
        print("Exploration AI's Reasoning:", explanation['narrative'])
        game_state = game_state_after
        time.sleep(1)

def start_adversarial_loop(model, arlc, em, adversary):
    print("\nStarting Adversarial Training Loop...")
    adversary.run_adversarial_training(arlc, em)
    print("\nAdversarial training complete. Model has self-corrected.")
    
def start_federated_learning_loop():
    config = Config()
    ckg = ConceptualKnowledgeGraph()
    global_model = ASREHModel().to(config.DEVICE)
    clients = [ASREHModel().to(config.DEVICE) for _ in range(config.NUM_CLIENTS)]
    for client in clients:
        client.load_state_dict(global_model.state_dict())
    print("Generating simulated client data...")
    client_data = [
        [{'state': np.random.rand(1, 10, 20), 'conceptual_features': np.random.rand(64), 'target': np.random.rand(64 * 64), 'domain': 'tetris'} for _ in range(10)]
        for _ in range(config.NUM_CLIENTS)
    ]
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(global_model.parameters(), lr=config.LEARNING_RATE)
    fl_learner = FederatedLearner(global_model, clients, criterion, optimizer, ckg) # Pass CKG
    fl_learner.run_federated_training(client_data)
    model_path = os.path.join(config.CHECKPOINT_DIR, "zenith_protocol_fl_final.pth")
    torch.save(global_model.state_dict(), model_path)
    print(f"\nFinal global model saved to {model_path}")

def start_meta_learning_loop():
    config = Config()
    ckg = ConceptualKnowledgeGraph()
    meta_model = ASREHModel().to(config.DEVICE)
    print("Preparing a set of diverse tasks for meta-training...")
    tasks = [
        {
            'domain': 'tetris',
            'train_data': [{'state': np.random.rand(1, 20, 10), 'conceptual_features': np.random.rand(64), 'target': np.random.rand(64 * 64), 'domain': 'tetris'} for _ in range(2)],
            'val_data': [{'state': np.random.rand(1, 20, 10), 'conceptual_features': np.random.rand(64), 'target': np.random.rand(64 * 64), 'domain': 'tetris'} for _ in range(2)]
        },
        {
            'domain': 'chess',
            'train_data': [{'state': np.random.rand(1, 8, 8), 'conceptual_features': np.random.rand(64), 'target': np.random.rand(64 * 64), 'domain': 'chess'} for _ in range(2)],
            'val_data': [{'state': np.random.rand(1, 8, 8), 'conceptual_features': np.random.rand(64), 'target': np.random.rand(64 * 64), 'domain': 'chess'} for _ in range(2)]
        }
    ]
    meta_learner = MetaLearner(meta_model, tasks, ckg) # Pass CKG
    meta_learner.run_meta_training()
    model_path = os.path.join(config.CHECKPOINT_DIR, "zenith_protocol_meta_final.pth")
    torch.save(meta_model.state_dict(), model_path)
    print(f"\nFinal meta-learned model saved to {model_path}")

def main():
    try:
        game_choice, mode_choice = select_game()
        if mode_choice == 'federated_learning':
            start_federated_learning_loop()
        elif mode_choice == 'meta_learning':
            start_meta_learning_loop()
        elif mode_choice == 'adversarial':
            game_env, model, arlc, em, adversary = load_game(game_choice)
            start_adversarial_loop(model, arlc, em, adversary)
        else:
            game_env, model, arlc, em, adversary = load_game(game_choice)
            if mode_choice == 'vs':
                start_game_loop(game_env, model, arlc, em, mode='vs')
            elif mode_choice == 'spectator':
                start_game_loop(game_env, model, arlc, em, mode='spectator')
            elif mode_choice == 'exploration':
                start_exploration_loop(game_env, model, arlc, em)
            else:
                print("Invalid mode choice. Exiting.")
                sys.exit()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
