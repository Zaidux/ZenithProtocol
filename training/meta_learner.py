# /src/training/meta_learner.py

import torch
import random
from typing import List, Dict, Any
from copy import deepcopy
from ..models.asreh_model import ASREHModel
from ..utils.config import Config
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph

class MetaLearner:
    """
    Implements a Model-Agnostic Meta-Learning (MAML) approach.
    It trains a model to find a good set of initial parameters that can
    be quickly adapted to new, unseen tasks (domains) with a few gradient steps.
    """
    def __init__(self, model: ASREHModel, tasks: List, ckg: ConceptualKnowledgeGraph):
        self.model = model
        self.ckg = ckg # New: CKG instance for storing meta-knowledge
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.META_LEARNING_RATE)
        self.criterion = torch.nn.MSELoss()
        self.tasks = tasks
        self.config = Config()

    def inner_loop_update(self, task_model: ASREHModel, task_data: List):
        """
        Performs the inner-loop update on a specific task.
        This adapts the task_model for a few gradient steps.
        """
        task_optimizer = torch.optim.Adam(task_model.parameters(), lr=self.config.INNER_LOOP_LR)
        task_model.train()

        for _ in range(self.config.INNER_LOOP_STEPS):
            for data_point in task_data:
                state, conceptual_features, target = data_point['state'], data_point['conceptual_features'], data_point['target']
                domain = data_point['domain']

                state_tensor = torch.tensor(state).unsqueeze(0).float().to(self.config.DEVICE)
                conceptual_tensor = torch.tensor(conceptual_features).unsqueeze(0).float().to(self.config.DEVICE)
                target_tensor = torch.tensor(target).unsqueeze(0).float().to(self.config.DEVICE)

                predicted_output, _, _ = task_model(state_tensor, conceptual_tensor, domain)
                loss = self.criterion(predicted_output, target_tensor)

                task_optimizer.zero_grad()
                loss.backward()
                task_optimizer.step()

    def outer_loop_update(self, adapted_models: List[ASREHModel], tasks: List):
        """
        Performs the outer-loop update on the main meta-model.
        This updates the model's parameters based on the performance of
        the adapted models across different tasks.
        """
        self.meta_optimizer.zero_grad()

        meta_loss = 0
        for i, adapted_model in enumerate(adapted_models):
            task_data = tasks[i]['val_data']
            task_domain = tasks[i]['domain']

            for data_point in task_data:
                state, conceptual_features, target = data_point['state'], data_point['conceptual_features'], data_point['target']

                state_tensor = torch.tensor(state).unsqueeze(0).float().to(self.config.DEVICE)
                conceptual_tensor = torch.tensor(conceptual_features).unsqueeze(0).float().to(self.config.DEVICE)
                target_tensor = torch.tensor(target).unsqueeze(0).float().to(self.config.DEVICE)

                predicted_output, _, _ = adapted_model(state_tensor, conceptual_tensor, task_domain)
                meta_loss += self.criterion(predicted_output, target_tensor)
        
        # New: Log meta-knowledge to the CKG
        self.ckg.add_node("Meta-Training", {"type": "training_phase", "description": "The model has learned how to learn."})
        self.ckg.add_edge("ASREHModel", "Meta-Training", "PARTICIPATED_IN")
        self.ckg.add_node(f"Meta-Loss_{meta_loss.item():.2f}", {"type": "metric", "value": meta_loss.item()})
        self.ckg.add_edge("Meta-Training", f"Meta-Loss_{meta_loss.item():.2f}", "RESULTED_IN")


        meta_loss.backward()
        self.meta_optimizer.step()
        print(f"Meta-loss: {meta_loss.item():.4f}")

    def run_meta_training(self):
        """
        Runs the full meta-training loop.
        """
        print(f"Starting Meta-Training for {self.config.META_TRAINING_EPOCHS} epochs.")

        for epoch in range(self.config.META_TRAINING_EPOCHS):
            print(f"\n--- Meta-Training Epoch {epoch + 1}/{self.config.META_TRAINING_EPOCHS} ---")

            sampled_tasks = random.sample(self.tasks, self.config.NUM_META_TASKS)

            adapted_models = []
            for task in sampled_tasks:
                task_model = deepcopy(self.model)
                self.inner_loop_update(task_model, task['train_data'])
                adapted_models.append(task_model)

            self.outer_loop_update(adapted_models, sampled_tasks)

        print("\nMeta-Training complete. The model is now ready for rapid adaptation.")
