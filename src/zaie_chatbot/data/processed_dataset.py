import json
import os
import re

# --- Step 1: Define File Paths ---
# Set the paths for our input and output files.
# We'll assume the conceptual_properties.json is in the same directory.
# The raw data will be a placeholder for now, but the structure is defined.

CONCEPTUAL_PROPERTIES_FILE = 'conceptual_properties.json'
RAW_CONVERSATIONAL_DATA_FILE = 'raw_conversational_data.json' # Placeholder
PROCESSED_DATASET_FILE = 'processed_zaie_dataset.json'

# --- Step 2: Load Data ---
def load_json_file(filepath):
    """Loads a JSON file and returns its content."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{filepath}'.")
        return None

# Placeholder for raw conversational data.
# In a real-world scenario, this would be a large dataset of dialogues.
# For now, we'll use a small, structured example.
def get_raw_data():
    """Returns a placeholder raw conversational dataset."""
    return [
        {"sentence": "He placed the book on the table because it was dusty."},
        {"sentence": "The robot moved the box."},
        {"sentence": "The dog barked at the mailman."},
        {"sentence": "I am working on the Zenith Project today."}
    ]

# --- Step 3: The Core Processing Logic ---
def process_sentence(sentence, conceptual_properties):
    """
    Processes a single sentence, identifies conceptual elements, and
    fuses them with the text.

    This is the heart of the "Understanding is Key" principle.
    It identifies and connects conceptual elements to the raw text.
    """
    processed_entry = {
        "original_text": sentence,
        "conceptual_features": {
            "agent": [],
            "object": [],
            "action": [],
            "motion": [],
            "bridge": [],
            "properties": []
        }
    }
    
    # Simple tokenization by splitting the sentence into words.
    tokens = re.findall(r'\b\w+\b', sentence.lower())
    
    # Iterate through each conceptual element type (agent, object, etc.)
    for element_type, element_data in conceptual_properties['conceptual_elements'].items():
        for example in element_data['examples']:
            example_sentence = example['sentence'].lower()
            
            # Use a simple pattern match for now.
            # In a more advanced version, we would use NLP models for this.
            for key, value in example['breakdown'].items():
                if key.lower() in sentence.lower():
                    # Check for whole words to avoid partial matches
                    if re.search(r'\b' + re.escape(key.lower()) + r'\b', sentence.lower()):
                        
                        # Add the found conceptual element to our entry
                        found_element = {
                            "word": key,
                            "type": value['element'],
                            "attributes": value.get('attributes', []),
                            "properties": value.get('properties', []),
                            "purpose": value.get('purpose', "")
                        }
                        processed_entry['conceptual_features'][element_type].append(found_element)
    
    # --- Step 4: The "Why" - Causal Connection Logic ---
    # This is a key part of our "connection" ability.
    # We will look for bridge words and their associated reasons.
    if 'because' in sentence.lower():
        # A simple check for the "because" bridge.
        # This will be refined in later phases to be more robust.
        parts = sentence.lower().split('because', 1)
        if len(parts) == 2:
            action_clause = parts[0].strip()
            reason_clause = parts[1].strip()
            
            # Find the action in the action clause
            for action_data in processed_entry['conceptual_features']['action']:
                if action_data['word'] in action_clause:
                    action_data['causal_reason'] = reason_clause
    
    return processed_entry

# --- Step 5: Main Function to Run the Script ---
def create_processed_dataset():
    """Main function to orchestrate the data processing pipeline."""
    # 1. Load the conceptual knowledge base.
    conceptual_properties = load_json_file(CONCEPTUAL_PROPERTIES_FILE)
    if conceptual_properties is None:
        return

    # 2. Get the raw conversational data.
    raw_data = get_raw_data()

    # 3. Process each entry and create the final dataset.
    processed_dataset = []
    for entry in raw_data:
        processed_entry = process_sentence(entry['sentence'], conceptual_properties)
        processed_dataset.append(processed_entry)
    
    # 4. Save the final dataset.
    with open(PROCESSED_DATASET_FILE, 'w') as f:
        json.dump(processed_dataset, f, indent=4)
    
    print(f"Successfully created processed dataset at '{PROCESSED_DATASET_FILE}'.")
    print(f"Total entries processed: {len(processed_dataset)}")

# Run the script
if __name__ == "__main__":
    create_processed_dataset()

