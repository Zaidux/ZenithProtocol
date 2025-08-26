## Adaptive Self-Regulating Explainable Hybrid (ASREH) AI

# Project Overview 🧠
This project, named the Zenith Protocol, is an innovative approach to building a generalizable, explainable AI model. Instead of training a separate model for each task, the Zenith Protocol leverages a single, unified architecture—the Adaptive Self-Regulating Explainable Hybrid (ASREH) Model—to learn and master multiple, disparate domains. We demonstrate this capability by training the model to play two completely different games: Chess and Tetris.
The core of the Zenith Protocol is the belief that true intelligence lies in understanding, not just memorization. The ASREH model achieves this through a novel combination of architectural components.

# Phase 1: The Foundation 🏗️
This phase establishes the foundational architecture by implementing the core of the ASREH model for a single domain: Tetris. We will build the essential components and prove the core hypothesis that the model can learn and reason based on both visual and conceptual data.

Key Components Implemented in Phase 1:
 * Conceptual Layer or The Self supervised world model(SSWM): The model is trained on a set of conceptual features for Tetris, including gaps, max_height, and board_fullness. The Conceptual Attention Layer processes this abstract knowledge, which is then fused with visual data from the Tetris board.

 * Adaptive-Reinforced Learning Controller (ARLC): The ARLC guides the training by providing a conceptual reward signal. It rewards moves that lead to a "better" board state, such as minimizing gaps. This teaches the model to optimize for high-level, strategic goals.

 * Explainability Module (EM): The EM provides transparency by analyzing the model's internal state. It generates human-readable explanations for the model's decisions, stating what conceptual goals were driving a move.

# Phase 2: The Split Mind (SPM) ☯️
In this phase, we will introduce the second domain, Chess, to implement the Split Mind architecture. The model will be trained on both Tetris and Chess simultaneously. This phase will require significant upgrades to the data loaders and the training loop to demonstrate the model's ability to learn across disparate domains using a single shared core.

# Phase 3: Hyper-Conceptual Thinking (HCT) 💡
The final phase focuses on implementing Hyper-Conceptual Thinking (HCT). This is a dynamic, advanced state of the Conceptual Attention Layer that allows the AI to form and apply novel, abstract concepts. This capability moves the model from simple reasoning to genuine creative problem-solving and will be activated by a unique "discovery bonus" in the ARLC's reward function.

# Phase 4: Autonomous Domain Exploration 🗺️
This phase focuses on making the Zenith Protocol a truly autonomous and self-improving system. We've enhanced the model with new capabilities that allow it to understand new environments on its own, optimize its performance, and provide deeper, more insightful explanations.

Key Components Implemented in Phase 4:
 * **Mixture of Experts (MoE)**: The model's architecture has been upgraded to include a **MoE** layer. The shared encoder now acts as a router, distributing the workload to specialized expert networks. This makes the system more scalable and energy-efficient.

 * **Energy of Movement (EoM)**: A new reward signal, the **Energy of Movement**, has been integrated into the ARLC. This bonus incentivizes moves that cause a significant and valuable conceptual shift in the model's internal state. It encourages the model to seek out strategic, rather than just tactical, changes.

 * **Dynamic Quantization**: The model is now capable of performing **dynamic quantization**—it can automatically compress its own weights to increase inference speed when a bottleneck is detected. A human can still override this function, as per the original design.

 * **Spatial-Semantic World Model (SSWM)**: We introduced a new component, the **SSWM**, which allows the model to predict the future state of the game based on hypothetical moves. This enables "what-if" scenario analysis, moving beyond reactive play to proactive strategic planning.

 * **Strategic Planner**: The ARLC is now enhanced with a **Strategic Planner** module. This allows the model to select and pursue high-level goals (e.g., "center control" in chess or "minimize gaps" in Tetris) and adjust its tactical decisions accordingly.

 * **Full Autonomous Exploration**: The HCT and ARLC modules now work together to enable the model to enter and learn from new, unknown domains without explicit human guidance. This is achieved through a "surprise bonus" that encourages exploration and a self-evolving conceptual knowledge base.

# Phase 5: Meta-Capabilities and On-Device Deployment 🌐
This phase elevates the Zenith Protocol's intelligence by focusing on meta-level improvements: learning how to learn and scaling to real-world applications with privacy in mind.

Key Components Implemented in Phase 5:
 * **Federated Learning and On-Device Deployment**: We've integrated a federated learning pipeline. This enables the model to be trained on decentralized data directly from user devices. The process aggregates model updates without ever exposing private gameplay data, addressing scalability and privacy concerns.

 * **Meta-Learning for Rapid Adaptation**: A new meta-learning module allows the model to learn a good set of initial parameters that can be rapidly fine-tuned for new, unknown domains. Instead of a long exploration process, the model can now adapt to a new game from just a few examples.

# Technologies Used 💻
 * PyTorch: The primary deep learning framework.
 * NumPy & Pillow: For efficient data manipulation and image processing.
 * Python-Chess: A powerful library for handling all Chess-related logic (introduced in Phase 2).
 * Pygame: Used for the game environments and real-time visualization of the model's performance (optional for training but good for demos).
 * Websockets & FastAPI: Future-proofing for a live, web-based demonstration of the project.

This project is a first step toward an AI that not only excels at diverse tasks but can also articulate its reasoning, paving the way for more robust and trustworthy artificial intelligence.

## Licensing

The Zenith Protocol is dual-licensed under the **AGPLv3 (Open-Source)** and a **Commercial License**.

* **For the community and non-commercial use**, we offer our project under the AGPLv3. This license is for individuals and organizations who are committed to the principles of open-source software and will also make their full source code available if they use the Zenith Protocol in a network service.
* **For commercial use**, especially in a proprietary application, please refer to the `COMMERCIAL-LICENSE.md` file for details on how to obtain a commercial license.

**Note:** The AGPLv3 version of this project explicitly prohibits its use in certain high-stakes domains, including military, medical, and legal applications. Please review the `LICENSE` file for the full terms and conditions.
