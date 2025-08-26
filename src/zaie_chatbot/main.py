### `main.py`

# This file will be the entry point for the user to interact with the trained chatbot. It will load the saved model and provide a simple command-line interface.

import torch
import json
import re

from zaie_chatbot.model.zaie_model import ZAIEModel
from zaie_chatbot.data.processed_dataset import create_processed_dataset # We'll need this to get the vocab
# from your_utils_folder import SomeUtilFunctions # Placeholder for utils

def run_chatbot():
    """
    Main function to run the ZAIE chatbot.
    """
    print("Loading ZAIE Chatbot...")

    # --- 1. Load Vocab and Model ---
    try:
        # We need to recreate the vocab to map tokens back to words
        create_processed_dataset()
        with open('zaie_chatbot/data/processed_zaie_dataset.json', 'r') as f:
            data = json.load(f)
            
        all_words = set()
        for entry in data:
            all_words.update(entry['original_text'].lower().split())
        vocab = {word: i for i, word in enumerate(sorted(list(all_words)))}
        vocab_size = len(vocab)
        idx_to_word = {i: word for word, i in vocab.items()}
        
    except FileNotFoundError:
        print("Error: Processed data or vocab not found. Please run 'python train_zaie.py' first.")
        return

    # Hyperparameters must match the training script
    hidden_dim = 256
    conceptual_embedding_size = 3 # Matches the size in train_zaie.py
    num_experts = 2
    num_heads = 4
    num_layers = 2
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ZAIEModel(vocab_size, conceptual_embedding_size, hidden_dim,
                      num_experts, num_heads, num_layers).to(device)
    
    # Load the trained model weights
    try:
        model.load_state_dict(torch.load('zaie_chatbot/model/zaie_model.pth'))
        model.eval()
    except FileNotFoundError:
        print("Error: Trained model not found. Please run 'python train_zaie.py' first.")
        return

    print("ZAIE Chatbot loaded! Type 'exit' to quit.")
    
    # --- 2. Chat Loop ---
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        # Simple conceptual feature extraction for inference
        conceptual_input = [1 if 'he' in user_input.lower() or 'she' in user_input.lower() or 'i ' in user_input.lower() else 0,
                            1 if 'book' in user_input.lower() or 'table' in user_input.lower() else 0,
                            1 if 'place' in user_input.lower() or 'move' in user_input.lower() else 0]
        
        conceptual_input_tensor = torch.tensor(conceptual_input, dtype=torch.float).unsqueeze(0).to(device)

        # Tokenize user input
        input_tokens = [vocab.get(word, 0) for word in user_input.lower().split()]
        input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)

        # Autoregressive generation
        generated_tokens = []
        with torch.no_grad():
            for _ in range(50): # Generate a max of 50 tokens
                output_logits, _ = model(input_tensor, conceptual_input_tensor)
                
                # Get the last token's logits
                last_token_logits = output_logits[0, -1, :]
                
                # Simple sampling
                probs = F.softmax(last_token_logits, dim=-1)
                next_token_id = torch.argmax(probs).item()
                
                if next_token_id == 0: # Assuming 0 is padding/end of sequence
                    break
                
                generated_tokens.append(next_token_id)
                
                # Append the generated token to the input for the next step
                input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_id]], device=device)], dim=1)
                
        # Decode the tokens back to words
        response_words = [idx_to_word.get(token_id, '[UNK]') for token_id in generated_tokens]
        response = " ".join(response_words)
        
        print(f"ZAIE: {response}")

if __name__ == "__main__":
    run_chatbot()
