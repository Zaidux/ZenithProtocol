import json

def get_raw_conversational_data():
    """
    This function acts as a placeholder for a real-world dataset.
    It returns a list of dictionaries, where each dictionary represents
    a single conversational entry.
    
    The 'text' field contains the raw, unstructured conversational input.
    In later phases, we will add 'agent_response' or 'label' fields
    for supervised fine-tuning.
    """
    
    # This is a sample of the raw conversational data.
    # We will use this to test our processing pipeline.
    # In a real-world application, this would be loaded from a large corpus.
    raw_data = [
        {"text": "He placed the book on the table because it was dusty."},
        {"text": "The robot moved the box."},
        {"text": "The dog barked at the mailman."},
        {"text": "I am working on the Zenith Project today."},
        {"text": "Why did the sky turn blue?"},
        {"text": "Can you help me write a poem about stars?"},
        {"text": "How do you define the word 'gravity'?"}
    ]
    
    return raw_data

def save_raw_data_to_file(data, filename="raw_conversational_data.json"):
    """
    Saves the raw conversational data to a JSON file for later use.
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Raw data saved to '{filename}'.")

if __name__ == "__main__":
    # Example of how to use this script.
    # We can run this to generate the raw data file our processed_dataset.py
    # script will use as input.
    raw_dataset = get_raw_conversational_data()
    save_raw_data_to_file(raw_dataset)

