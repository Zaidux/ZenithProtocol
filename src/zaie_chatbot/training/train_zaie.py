import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import os

# Assuming all models and data scripts are in src/
from zaie_chatbot.model.zaie_model import ZAIEModel
from zaie_chatbot.data.processed_dataset import create_processed_dataset
# from your_utils_folder import SomeUtilFunctions # Placeholder for utils


def train_zaie_model(epochs=100, learning_rate=1e-4, batch_size=4):
    """
    The main training function for the ZAIE chatbot.
    """
    print("Starting Phase 1 training for the ZAIE Chatbot...")

    # --- 1. Prepare Data ---
    # We call our data processing script to get the dataset.
    # Note: For a real project, we would not re-create the dataset on every run.
    create_processed_dataset()
    try:
        with open('zaie_chatbot/data/processed_zaie_dataset.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Processed dataset not found. Please check file path.")
        return

    # For Phase 1, we'll create a simple vocabulary.
    # This would be a more complex tokenizer in a production system.
    all_words = set()
    for entry in data:
        all_words.update(entry['original_text'].lower().split())
    vocab = {word: i for i, word in enumerate(sorted(list(all_words)))}
    vocab_size = len(vocab)
    
    # --- 2. Convert Data to Tensors ---
    text_inputs = []
    conceptual_inputs = []
    target_outputs = []

    for entry in data:
        tokens = [vocab.get(word, 0) for word in entry['original_text'].lower().split()]
        if not tokens:
            continue
        
        # We'll use a simplified conceptual vector for now
        # A more complex system would create a unique vector for each entry
        # based on the identified conceptual features.
        conceptual_vector = [1 if entry['conceptual_features']['agent'] else 0,
                             1 if entry['conceptual_features']['object'] else 0,
                             1 if entry['conceptual_features']['action'] else 0]
        
        # We'll use a simple next-token prediction task for training
        text_input = torch.tensor(tokens[:-1], dtype=torch.long)
        target_output = torch.tensor(tokens[1:], dtype=torch.long)

        # Pad sequences to a fixed length for batching
        text_inputs.append(text_input)
        conceptual_inputs.append(torch.tensor(conceptual_vector, dtype=torch.float).unsqueeze(0))
        target_outputs.append(target_output)

    text_inputs = pad_sequence(text_inputs, batch_first=True, padding_value=0)
    conceptual_inputs = torch.cat(conceptual_inputs, dim=0)
    target_outputs = pad_sequence(target_outputs, batch_first=True, padding_value=-100) # -100 is ignored by CrossEntropyLoss

    dataset = TensorDataset(text_inputs, conceptual_inputs, target_outputs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- 3. Initialize Model and Optimizer ---
    # Hyperparameters
    hidden_dim = 256
    conceptual_embedding_size = len(conceptual_vector)
    num_experts = 2 # From Phase 2
    num_heads = 4
    num_layers = 2
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ZAIEModel(vocab_size, conceptual_embedding_size, hidden_dim,
                      num_experts, num_heads, num_layers).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # --- 4. Training Loop ---
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (text_batch, conceptual_batch, target_batch) in enumerate(dataloader):
            text_batch, conceptual_batch, target_batch = text_batch.to(device), conceptual_batch.to(device), target_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output_logits, _ = model(text_batch, conceptual_batch)
            
            # Reshape for loss calculation
            loss = criterion(output_logits.view(-1, vocab_size), target_batch.view(-1))
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    print("Training complete!")
    # Save the trained model
    torch.save(model.state_dict(), 'zaie_chatbot/model/zaie_model.pth')
    print("Model saved to zaie_chatbot/model/zaie_model.pth")


