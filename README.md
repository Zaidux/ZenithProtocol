## Adaptive Self-Regulating Explainable Hybrid (ASREH) AI

# Project Overview 🧠
This project, named the Zenith Protocol, is an innovative approach to building a generalizable, explainable AI model. Instead of training a separate model for each task, the Zenith Protocol leverages a single, unified architecture—the Adaptive Self-Regulating Explainable Hybrid (ASREH) Model—to learn and master multiple, disparate domains. We demonstrate this capability by training the model to play two completely different games: Chess and Tetris.
The core of the Zenith Protocol is the belief that true intelligence lies in understanding, not just memorization. The ASREH model achieves this through a novel combination of architectural components.

# Phase 1: The Foundation 🏗️
This phase establishes the foundational architecture by implementing the core of the ASREH model for a single domain: Tetris. We will build the essential components and prove the core hypothesis that the model can learn and reason based on both visual and conceptual data.

Key Components Implemented in Phase 1:
 * Conceptual Layer: The model is trained on a set of conceptual features for Tetris, including gaps, max_height, and board_fullness. The Conceptual Attention Layer processes this abstract knowledge, which is then fused with visual data from the Tetris board.

 * Action-Reinforced Learning Controller (ARLC): The ARLC guides the training by providing a conceptual reward signal. It rewards moves that lead to a "better" board state, such as minimizing gaps. This teaches the model to optimize for high-level, strategic goals.

 * Explainability Module (EM): The EM provides transparency by analyzing the model's internal state. It generates human-readable explanations for the model's decisions, stating what conceptual goals were driving a move.

# Phase 2: The Split Mind (SPM) ☯️
In this phase, we will introduce the second domain, Chess, to implement the Split Mind architecture. The model will be trained on both Tetris and Chess simultaneously. This phase will require significant upgrades to the data loaders and the training loop to demonstrate the model's ability to learn across disparate domains using a single shared core.

# Phase 3: Hyper-Conceptual Thinking (HCT) 💡
The final phase focuses on implementing Hyper-Conceptual Thinking (HCT). This is a dynamic, advanced state of the Conceptual Attention Layer that allows the AI to form and apply novel, abstract concepts. This capability moves the model from simple reasoning to genuine creative problem-solving and will be activated by a unique "discovery bonus" in the ARLC's reward function.

Technologies Used 💻
 * PyTorch: The primary deep learning framework.
 * NumPy & Pillow: For efficient data manipulation and image processing.
 * Python-Chess: A powerful library for handling all Chess-related logic (introduced in Phase 2).
 * Pygame: Used for the game environments and real-time visualization of the model's performance (optional for training but good for demos).
 * Websockets & FastAPI: Future-proofing for a live, web-based demonstration of the project.

This project is a first step toward an AI that not only excels at diverse tasks but can also articulate its reasoning, paving the way for more robust and trustworthy artificial intelligence.