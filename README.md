## Adaptive Self-Regulating Explainable Hybrid (ASREH) AI

# Project Overview 🧠
This project, named the Zenith Protocol, is an innovative approach to building a generalizable, explainable AI model.

Instead of training a separate model for each task, the Zenith Protocol leverages a single, unified architecture—the Adaptive Self-Regulating Explainable Hybrid (ASREH) Model—to learn and master multiple, disparate domains.

We demonstrate this capability by training the model to play two completely different games: Chess and Tetris.

The core of the Zenith Protocol is the belief that true intelligence lies in understanding, not just memorization. The ASREH model achieves this through a novel combination of architectural components:

# 1. Split Mind (SPM) ☯️
The Split Mind architecture uses a single shared core to process visual data from both games. The model learns common patterns, shapes, and spatial relationships that are relevant to both Tetris boards and Chessboards. This shared learning foundation allows for cross-domain generalization, enabling the model to apply knowledge from one game to the other.

# 2. Conceptual Layer and Hyper-Conceptual Thinking (HCT) 💡
Beyond visual input, the model is trained on a set of conceptual features. For Chess, these include abstract concepts like material advantage, king safety, and center control. For Tetris, they include board fullness, number of gaps, and lines cleared. The Conceptual Layer processes this abstract knowledge, which is then fused with the visual data in the Hyper-Conceptual Thinking (HCT) layer. This fusion enables the model to make decisions based on both what it sees and what it understands.

# 3. Action-Reinforced Learning Controller (ARLC) 🎯
The ARLC is a custom training loop that goes beyond simple supervised learning. It acts as a guide, providing a conceptual reward signal that reinforces moves leading to "better" states. For example, a move that improves king safety in Chess or reduces the number of gaps in Tetris is rewarded more heavily. This encourages the model to learn and optimize for high-level, strategic goals rather than just rote patterns.

# 4. Explainability Module (EM) 💬
The EM is designed to provide transparency and build trust. By analyzing the model's internal state—specifically the attention weights and the conceptual embeddings—the EM generates human-readable explanations for every move. It can tell us what part of the board the model was focusing on and what conceptual goals were driving its decision. This is a crucial feature for ensuring accountability and debugging the model's behavior.

Technologies Used 💻
 * PyTorch: The primary deep learning framework.
 * Python-Chess: A powerful library for handling all Chess-related logic.
 * Pygame: Used for the game environments and real-time visualization of the model's performance.
 * NumPy & Pillow: For efficient data manipulation and image processing.
 * Websockets & FastAPI: Future-proofing for a live, web-based demonstration of the project.
This project is a first step toward an AI that not only excels at diverse tasks but can also articulate its reasoning, paving the way for more robust and trustworthy artificial intelligence.
