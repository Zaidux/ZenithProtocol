# /src/training/zenith_training_orchestrator.py

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from ..models.asreh_model import ASREHModel
from ..models.arlc_controller import ARLCController
from ..models.explainability_module import ExplainabilityModule
from ..models.sswm import SSWM
from ..models.hyper_conceptual_thinking import ConceptDiscoveryEngine
from ..models.strategic_planner import StrategicPlanner
from ..models.adversarial_module import AdversarialModule # New Import
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph # New Import
from ..web_access.web_access import WebAccess
from ..training.meta_learner import MetaLearner # New Import
from ..data.tetris_generator import MemoryEfficientDataset
from ..data.chess_generator import ChessDataset
from ..utils.config import Config
from ..utils.dynamic_quantization import DynamicQuantization
import os
import itertools
import numpy as np

# Create a configuration object instance
config = Config()

class ZenithTrainingOrchestrator:
    def __init__(self):
        self.config = config
        self.ckg = ConceptualKnowledgeGraph()
        self.web_access = WebAccess(self.ckg)
        self.model = ASREHModel(
            in_channels=self.config.IN_CHANNELS,
            hct_dim=self.config.HCT_DIM,
            num_experts=self.config.NUM_EXPERTS,
            ckg=self.ckg,
            web_access=self.web_access
        ).to(self.config.DEVICE)
        
        self.sswm = SSWM(
            input_dim=self.config.HCT_DIM,
            hidden_dim=self.config.HCT_DIM,
            ckg=self.ckg,
            web_access=self.web_access
        ).to(self.config.DEVICE)

        self.strategic_planner = StrategicPlanner(model=self.model, ckg=self.ckg)
        
        # ARLC now gets the CKG and WebAccess
        self.arlc = ARLCController(
            strategic_planner=self.strategic_planner,
            sswm=self.sswm,
            ckg=self.ckg,
            web_access=self.web_access
        )
        self.em = ExplainabilityModule(
            model=self.model,
            sswm=self.sswm,
            ckg=self.ckg
        )
        self.adversary = AdversarialModule(model=self.model)

        self.tetris_loader, self.chess_loader = self.get_dataloaders()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        self.tetris_wm_loss_fn = nn.BCELoss()
        self.chess_policy_loss_fn = nn.CrossEntropyLoss()
        
    def get_dataloaders(self):
        print("Initializing Tetris and Chess datasets...")
        tetris_dataset = MemoryEfficientDataset(size=self.config.TETRIS_DATA_SIZE)
        chess_dataset = ChessDataset(size=self.config.CHESS_DATA_SIZE)

        tetris_loader = DataLoader(tetris_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() // 2 or 1)
        chess_loader = DataLoader(chess_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() // 2 or 1)
        return tetris_loader, chess_loader

    def phase4_autonomous_exploration(self):
        print("Starting Phase 4: Autonomous Exploration training loop...")
        last_fused_representation = None
        tetris_iter = itertools.cycle(self.tetris_loader)
        chess_iter = itertools.cycle(self.chess_loader)
        num_batches = len(self.tetris_loader) + len(self.chess_loader)

        for epoch in range(self.config.NUM_EPOCHS):
            total_tetris_loss = 0
            total_chess_loss = 0
            for i in range(num_batches):
                if i % 2 == 0:
                    state_before_img, board_after_img, _, conceptual_features = next(tetris_iter)
                    domain = 'tetris'
                    state_before_img = state_before_img.to(self.config.DEVICE)
                    board_after_img = board_after_img.to(self.config.DEVICE)
                    conceptual_features = conceptual_features.to(self.config.DEVICE)
                    predicted_board, fused_representation, moe_loss = self.model(state_before_img, conceptual_features, domain)
                    main_loss = self.tetris_wm_loss_fn(predicted_board, board_after_img)
                    total_loss = main_loss + moe_loss
                    total_tetris_loss += total_loss.item()
                else:
                    state_before_img, move_idx, conceptual_features = next(chess_iter)
                    domain = 'chess'
                    state_before_img = state_before_img.to(self.config.DEVICE)
                    move_idx = move_idx.to(self.config.DEVICE)
                    conceptual_features = conceptual_features.to(self.config.DEVICE)
                    predicted_move_logits, fused_representation, moe_loss = self.model(state_before_img, conceptual_features, domain)
                    main_loss = self.chess_policy_loss_fn(predicted_move_logits, move_idx)
                    total_loss = main_loss + moe_loss
                    total_chess_loss += total_loss.item()

                if last_fused_representation is not None:
                    self.arlc.calculate_eom_bonus(last_fused_representation, fused_representation)
                last_fused_representation = fused_representation.detach().clone()

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                if (i + 1) % self.config.LOG_INTERVAL == 0:
                    print(f"Epoch [{epoch+1}/{self.config.NUM_EPOCHS}], Step [{i+1}/{num_batches}], "
                          f"Domain: {domain.capitalize()}, Loss: {total_loss.item():.4f}")
            
            if DynamicQuantization.should_quantize(epoch, total_tetris_loss, total_chess_loss):
                print("Applying dynamic quantization to the model...")
                self.model = DynamicQuantization.quantize_model(self.model)
                print("Model quantized. Continuing training with a faster model.")

            avg_tetris_loss = total_tetris_loss / len(self.tetris_loader)
            avg_chess_loss = total_chess_loss / len(self.chess_loader)
            print(f"Epoch {epoch+1} finished. Avg Tetris Loss: {avg_tetris_loss:.4f}, Avg Chess Loss: {avg_chess_loss:.4f}")

    def phase5_meta_learning(self):
        print("\nStarting Phase 5: Cross-Domain and Meta-Learning...")
        # Create a simple task list for the meta-learner
        tasks = [
            {'domain': 'tetris', 'train_data': list(self.tetris_loader), 'val_data': list(self.tetris_loader)},
            {'domain': 'chess', 'train_data': list(self.chess_loader), 'val_data': list(self.chess_loader)}
        ]
        meta_learner = MetaLearner(self.model, tasks, self.ckg)
        meta_learner.run_meta_training()
    
    def phase6_adversarial_self_correction(self):
        print("\nStarting Phase 6: Adversarial and Self-Correctional Training...")
        self.adversary.run_adversarial_training(self.arlc, self.em)
        
    def run_all_phases(self):
        print(f"Using device: {self.config.DEVICE}")
        self.phase4_autonomous_exploration()
        self.phase5_meta_learning()
        self.phase6_adversarial_self_correction()
        print("All training phases finished!")
        torch.save(self.model.state_dict(), os.path.join(self.config.CHECKPOINT_DIR, "zenith_protocol_final.pth"))
        print(f"Final model saved to {self.config.CHECKPOINT_DIR}/zenith_protocol_final.pth")

if __name__ == '__main__':
    orchestrator = ZenithTrainingOrchestrator()
    orchestrator.run_all_phases()
