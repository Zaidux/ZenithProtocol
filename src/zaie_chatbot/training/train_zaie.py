import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import json
import os

# Assuming all models and data scripts are in src/
from zaie_chatbot.model.zaie_model import ZAIEModel

def load_all_data(root_dir='datasets'):
    """
    Recursively loads all .json and .jsonl files from a given directory.
    """
    all_data = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.jsonl'):
                filepath = os.path.join(dirpath, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        all_data.append(json.loads(line))
            elif filename.endswith('.json'):
                filepath = os.path.join(dirpath, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    all_data.extend(json.load(f))
    return all_data

def train_zaie_model(epochs=100, learning_rate=1e-4, batch_size=4):
    """
    The main training function for the ZAIE chatbot.
    """
    print("Starting Phase 1 training for the ZAIE Chatbot...")

    # --- 1. Prepare Data ---
    # Load all data from the 'datasets' directory
    print("Loading data from all files in the 'datasets' directory...")
    data = load_all_data()
    print(f"Found {len(data)} data entries.")

    if not data:
        print("No data found. Please check the 'datasets' directory.")
        return

    # For Phase 1, we'll create a simple vocabulary.
    all_words = set()
    for entry in data:
        # We handle both 'prompt' and 'code_snippet' keys
        text = entry.get('prompt') or entry.get('code_snippet')
        if text:
            all_words.update(text.lower().split())

    vocab = {word: i for i, word in enumerate(sorted(list(all_words)))}
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    # --- 2. Convert Data to Tensors ---
    text_inputs = []
    conceptual_inputs = []
    target_outputs = []

    for entry in data:
        text = entry.get('prompt') or entry.get('code_snippet')
        if not text:
            continue
        
        tokens = [vocab.get(word, 0) for word in text.lower().split()]
        if not tokens:
            continue

        # Use a simplified conceptual vector based on the presence of keys
        conceptual_vector = [1 if 'purpose' in entry else 0,
                             1 if 'conceptual_breakdown' in entry else 0,
                             1 if 'causal_chain' in entry else 0,
                             1 if 'explanation_module' in entry else 0]

        if len(tokens) > 1:
            text_input = torch.tensor(tokens[:-1], dtype=torch.long)
            target_output = torch.tensor(tokens[1:], dtype=torch.long)

            text_inputs.append(text_input)
            conceptual_inputs.append(torch.tensor(conceptual_vector, dtype=torch.float).unsqueeze(0))
            target_outputs.append(target_output)

    if not text_inputs:
        print("No valid data entries found for training.")
        return

    text_inputs = pad_sequence(text_inputs, batch_first=True, padding_value=0)
    conceptual_inputs = torch.cat(conceptual_inputs, dim=0)
    target_outputs = pad_sequence(target_outputs, batch_first=True, padding_value=-100) # -100 is ignored by CrossEntropyLoss

    dataset = TensorDataset(text_inputs, conceptual_inputs, target_outputs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # --- 3. Initialize Model and Optimizer ---
    # The rest of your code remains the same.
    hidden_dim = 256
    conceptual_embedding_size = conceptual_inputs.shape[1]
    num_experts = 2
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

            output_logits, _ = model(text_batch, conceptual_batch)

            loss = criterion(output_logits.view(-1, vocab_size), target_batch.view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    print("Training complete!")
    torch.save(model.state_dict(), 'zaie_chatbot/model/zaie_model.pth')
    print("Model saved to zaie_chatbot/model/zaie_model.pth")

if __name__ == '__main__':
    train_zaie_model()
