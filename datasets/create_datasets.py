import os
import json

# This script creates the foundational 'datasets' folder and populates it with
# example data files for the Zenith Protocol. The data is structured
# according to the conceptual categories defined in the protocol.

def create_zenith_datasets():
    """
    Creates a 'datasets' directory with subfolders for chatbot and image models.
    It then populates these subfolders with example data in JSON Lines format.
    """
    # --- Step 1: Define the directory structure ---
    # The datasets folder is at the project root, separate from source code.
    base_dir = "datasets"
    chatbot_dir = os.path.join(base_dir, "chatbot")
    image_dir = os.path.join(base_dir, "image_generation")

    # --- Step 2: Create directories if they don't exist ---
    print(f"Creating directories: {chatbot_dir} and {image_dir}")
    os.makedirs(chatbot_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    print("Directories created successfully.")

    # --- Step 3: Define example data for chatbot models ---
    # This data includes the conceptual breakdown for the 'Zenith Protocol's
    # conceptual attention layer. Each line is a JSON object.
    chatbot_data = [
        {
            "prompt": "Why did the man run?",
            "conceptual_breakdown": {
                "Agent": "man",
                "Action": "run",
                "Reason": "to catch the bus",
                "Possibilities": ["can walk", "can run", "can drive"],
                "Properties": {"bus": "heavy", "man": "fast"}
            }
        },
        {
            "prompt": "How do you build a campfire?",
            "conceptual_breakdown": {
                "Agent": "you",
                "Action": "build",
                "Object": "campfire",
                "Cause": "to provide warmth and light",
                "Background": "outdoors",
                "Properties": {"wood": "flammable", "fire": "hot"}
            }
        },
        {
            "prompt": "Explain the five-step reasoning process.",
            "conceptual_breakdown": {
                "Agent": "protocol",
                "Action": "explain",
                "Object": "reasoning process",
                "Motion": "continuous",
                "Bridge": {"the five-step": "quantifier"}
            }
        }
    ]

    # --- Step 4: Write the chatbot data to a file ---
    chatbot_file_path = os.path.join(chatbot_dir, "qa_data.jsonl")
    with open(chatbot_file_path, 'w') as f:
        for entry in chatbot_data:
            f.write(json.dumps(entry) + '\n')
    print(f"Chatbot data file created at: {chatbot_file_path}")

    # --- Step 5: Define example data for image generation models ---
    # This data includes the new 'Background' and 'Color' categories.
    image_data = [
        {
            "image_id": "image_001",
            "prompt": "A young knight with striking red hair is fighting a dragon in a dark, mysterious forest.",
            "conceptual_breakdown": {
                "Agent": "young knight",
                "Action": "fighting",
                "Object": "dragon",
                "Background": {"environment": "forest", "quality": "dark, mysterious"},
                "Color": {"knight": {"hair": "red"}, "dragon": {"scales": "green"}}
            }
        },
        {
            "image_id": "image_002",
            "prompt": "A futuristic robot with a shiny blue finish is serving coffee on a bright, sunny rooftop.",
            "conceptual_breakdown": {
                "Agent": "futuristic robot",
                "Action": "serving",
                "Object": "coffee",
                "Background": {"environment": "rooftop", "quality": "bright, sunny"},
                "Color": {"robot": {"body": "blue"}}
            }
        }
    ]

    # --- Step 6: Write the image data to a file ---
    image_file_path = os.path.join(image_dir, "conceptual_image_data.jsonl")
    with open(image_file_path, 'w') as f:
        for entry in image_data:
            f.write(json.dumps(entry) + '\n')
    print(f"Image generation data file created at: {image_file_path}")

    print("\nDataset generation complete!")

# Run the function to create the datasets.
if __name__ == "__main__":
    create_zenith_datasets()

