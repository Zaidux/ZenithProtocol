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
from ..models.adversarial_module import AdversarialModule
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph
from ..web_access.web_access import WebAccess
from ..training.meta_learner import MetaLearner
from ..data.tetris_generator import MemoryEfficientDataset
from ..data.chess_generator import ChessDataset
from ..utils.config import Config
from ..utils.dynamic_quantization import DynamicQuantization
from ..attention.zenith_sparse_attention import ZenithSparseAttention
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
        
        # Initialize Zenith Sparse Attention
        self.zenith_attention = ZenithSparseAttention(
            dim=self.config.HCT_DIM,
            num_heads=self.config.ZENITH_NUM_HEADS,
            sparsity_threshold=self.config.ZENITH_SPARSITY_THRESHOLD,
            top_k_sparse=self.config.ZENITH_TOP_K_SPARSE,
            ckg_guidance=True,
            ckg=self.ckg
        ).to(self.config.DEVICE)
        
        self.model = ASREHModel(
            in_channels=self.config.IN_CHANNELS,
            hct_dim=self.config.HCT_DIM,
            num_experts=self.config.NUM_EXPERTS,
            ckg=self.ckg,
            web_access=self.web_access,
            zenith_attention=self.zenith_attention  # Pass zenith attention
        ).to(self.config.DEVICE)

        self.sswm = SSWM(
            input_dim=self.config.HCT_DIM,
            hidden_dim=self.config.HCT_DIM,
            ckg=self.ckg,
            web_access=self.web_access,
            zenith_attention=self.zenith_attention
        ).to(self.config.DEVICE)

        self.strategic_planner = StrategicPlanner(
            model=self.model, 
            ckg=self.ckg,
            zenith_attention=self.zenith_attention
        )

        self.arlc = ARLCController(
            strategic_planner=self.strategic_planner,
            sswm=self.sswm,
            ckg=self.ckg,
            web_access=self.web_access,
            zenith_attention=self.zenith_attention
        )
        
        self.em = ExplainabilityModule(
            model=self.model,
            sswm=self.sswm,
            ckg=self.ckg,
            zenith_attention=self.zenith_attention
        )
        
        self.adversary = AdversarialModule(
            model=self.model,
            zenith_attention=self.zenith_attention
        )

        self.tetris_loader, self.chess_loader = self.get_dataloaders()
        
        # Optimizer with sparse attention parameters
        zenith_params = list(self.zenith_attention.parameters())
        model_params = list(self.model.parameters())
        sswm_params = list(self.sswm.parameters())
        
        all_params = zenith_params + model_params + sswm_params
        self.optimizer = optim.Adam(all_params, lr=self.config.LEARNING_RATE)
        
        self.tetris_wm_loss_fn = nn.BCELoss()
        self.chess_policy_loss_fn = nn.CrossEntropyLoss()
        
        # Sparse attention monitoring
        self.attention_sparsity_history = []
        self.computational_savings = []

    def get_dataloaders(self):
        print("Initializing Tetris and Chess datasets with sparse attention optimization...")
        tetris_dataset = MemoryEfficientDataset(size=self.config.TETRIS_DATA_SIZE)
        chess_dataset = ChessDataset(size=self.config.CHESS_DATA_SIZE)

        tetris_loader = DataLoader(
            tetris_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True, 
            num_workers=os.cpu_count() // 2 or 1,
            pin_memory=True  # Optimized for sparse attention
        )
        chess_loader = DataLoader(
            chess_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True, 
            num_workers=os.cpu_count() // 2 or 1,
            pin_memory=True
        )
        return tetris_loader, chess_loader

    def apply_sparse_attention_optimizations(self, epoch):
        """Dynamically adjust sparse attention parameters based on training progress"""
        if epoch > self.config.ZENITH_ADAPTATION_START_EPOCH:
            # Gradually increase sparsity as training progresses
            progress_ratio = min(1.0, epoch / self.config.NUM_EPOCHS)
            new_sparsity = self.config.ZENITH_SPARSITY_THRESHOLD * (0.5 + 0.5 * progress_ratio)
            new_top_k = max(8, int(self.config.ZENITH_TOP_K_SPARSE * (1.0 - 0.3 * progress_ratio)))
            
            self.zenith_attention.update_sparsity_params(
                sparsity_threshold=new_sparsity,
                top_k_sparse=new_top_k
            )
            
            print(f"Updated sparse attention: sparsity_threshold={new_sparsity:.3f}, top_k={new_top_k}")

    def monitor_sparse_attention(self, batch_idx, domain):
        """Monitor and log sparse attention efficiency"""
        if batch_idx % self.config.ZENITH_MONITOR_INTERVAL == 0:
            sparsity_stats = self.zenith_attention.get_sparsity_stats()
            self.attention_sparsity_history.append(sparsity_stats)
            
            computational_saving = (
                sparsity_stats['sparsity_ratio'] * 
                sparsity_stats['attention_blocks_pruned'] / 
                sparsity_stats['total_attention_blocks']
            )
            self.computational_savings.append(computational_saving)
            
            print(f"Zenith Sparse Attention Stats - "
                  f"Sparsity: {sparsity_stats['sparsity_ratio']:.3f}, "
                  f"Pruned: {sparsity_stats['attention_blocks_pruned']}/{sparsity_stats['total_attention_blocks']}, "
                  f"Savings: {computational_saving:.2%}")

    def phase4_autonomous_exploration(self):
        print("Starting Phase 4: Autonomous Exploration with Zenith Sparse Attention...")
        last_fused_representation = None
        tetris_iter = itertools.cycle(self.tetris_loader)
        chess_iter = itertools.cycle(self.chess_loader)
        num_batches = len(self.tetris_loader) + len(self.chess_loader)

        for epoch in range(self.config.NUM_EPOCHS):
            # Apply dynamic sparse attention optimizations
            self.apply_sparse_attention_optimizations(epoch)
            
            total_tetris_loss = 0
            total_chess_loss = 0
            zenith_sparse_loss = 0
            
            for i in range(num_batches):
                if i % 2 == 0:
                    # Tetris domain
                    state_before_img, board_after_img, _, conceptual_features = next(tetris_iter)
                    domain = 'tetris'
                    state_before_img = state_before_img.to(self.config.DEVICE)
                    board_after_img = board_after_img.to(self.config.DEVICE)
                    conceptual_features = conceptual_features.to(self.config.DEVICE)
                    
                    predicted_board, fused_representation, moe_loss = self.model(
                        state_before_img, conceptual_features, domain
                    )
                    main_loss = self.tetris_wm_loss_fn(predicted_board, board_after_img)
                    
                    # Add sparse attention regularization
                    sparse_reg = self.zenith_attention.get_sparsity_regularization()
                    total_loss = main_loss + moe_loss + self.config.ZENITH_SPARSITY_REG * sparse_reg
                    zenith_sparse_loss += sparse_reg.item()
                    
                    total_tetris_loss += total_loss.item()
                else:
                    # Chess domain
                    state_before_img, move_idx, conceptual_features = next(chess_iter)
                    domain = 'chess'
                    state_before_img = state_before_img.to(self.config.DEVICE)
                    move_idx = move_idx.to(self.config.DEVICE)
                    conceptual_features = conceptual_features.to(self.config.DEVICE)
                    
                    predicted_move_logits, fused_representation, moe_loss = self.model(
                        state_before_img, conceptual_features, domain
                    )
                    main_loss = self.chess_policy_loss_fn(predicted_move_logits, move_idx)
                    
                    # Add sparse attention regularization
                    sparse_reg = self.zenith_attention.get_sparsity_regularization()
                    total_loss = main_loss + moe_loss + self.config.ZENITH_SPARSITY_REG * sparse_reg
                    zenith_sparse_loss += sparse_reg.item()
                    
                    total_chess_loss += total_loss.item()

                # ARLC bonus calculation with sparse attention context
                if last_fused_representation is not None:
                    self.arlc.calculate_eom_bonus(
                        last_fused_representation, 
                        fused_representation,
                        self.zenith_attention.get_current_sparsity_pattern()
                    )
                last_fused_representation = fused_representation.detach().clone()

                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping for sparse attention stability
                torch.nn.utils.clip_grad_norm_(
                    self.zenith_attention.parameters(), 
                    self.config.ZENITH_GRAD_CLIP
                )
                
                self.optimizer.step()

                # Monitor sparse attention efficiency
                self.monitor_sparse_attention(i, domain)

                if (i + 1) % self.config.LOG_INTERVAL == 0:
                    current_sparsity = self.zenith_attention.get_sparsity_stats()['sparsity_ratio']
                    print(f"Epoch [{epoch+1}/{self.config.NUM_EPOCHS}], Step [{i+1}/{num_batches}], "
                          f"Domain: {domain.capitalize()}, Loss: {total_loss.item():.4f}, "
                          f"Sparsity: {current_sparsity:.3f}")

            # Dynamic quantization with sparse attention awareness
            if DynamicQuantization.should_quantize(
                epoch, 
                total_tetris_loss + total_chess_loss,
                self.zenith_attention.get_computational_efficiency()
            ):
                print("Applying dynamic quantization with sparse attention preservation...")
                self.model = DynamicQuantization.quantize_model(self.model)
                # Re-apply zenith attention to quantized model
                self.model.zenith_attention = self.zenith_attention
                print("Model quantized with preserved sparse attention patterns.")

            avg_tetris_loss = total_tetris_loss / len(self.tetris_loader)
            avg_chess_loss = total_chess_loss / len(self.chess_loader)
            avg_zenith_loss = zenith_sparse_loss / num_batches
            
            print(f"Epoch {epoch+1} finished. "
                  f"Avg Tetris Loss: {avg_tetris_loss:.4f}, "
                  f"Avg Chess Loss: {avg_chess_loss:.4f}, "
                  f"Avg Zenith Sparsity Loss: {avg_zenith_loss:.4f}")

    def phase5_meta_learning(self):
        print("\nStarting Phase 5: Cross-Domain and Meta-Learning with Sparse Attention...")
        # Create tasks with sparse attention context
        tasks = [
            {
                'domain': 'tetris', 
                'train_data': list(self.tetris_loader), 
                'val_data': list(self.tetris_loader),
                'sparse_attention': self.zenith_attention
            },
            {
                'domain': 'chess', 
                'train_data': list(self.chess_loader), 
                'val_data': list(self.chess_loader),
                'sparse_attention': self.zenith_attention
            }
        ]
        meta_learner = MetaLearner(
            self.model, 
            tasks, 
            self.ckg,
            self.zenith_attention
        )
        meta_learner.run_meta_training()

    def phase6_adversarial_self_correction(self):
        print("\nStarting Phase 6: Adversarial and Self-Correctional Training...")
        self.adversary.run_adversarial_training(self.arlc, self.em)

    def run_all_phases(self):
        print(f"Using device: {self.config.DEVICE}")
        print("Initializing Zenith Sparse Attention Training Orchestrator...")
        
        # Log zenith configuration
        zenith_config = {
            'num_heads': self.config.ZENITH_NUM_HEADS,
            'sparsity_threshold': self.config.ZENITH_SPARSITY_THRESHOLD,
            'top_k_sparse': self.config.ZENITH_TOP_K_SPARSE,
            'ckg_guidance': True
        }
        print(f"Zenith Sparse Attention Config: {zenith_config}")
        
        self.phase4_autonomous_exploration()
        self.phase5_meta_learning()
        self.phase6_adversarial_self_correction()
        
        # Final sparse attention statistics
        final_stats = self.zenith_attention.get_sparsity_stats()
        avg_savings = np.mean(self.computational_savings) if self.computational_savings else 0
        print(f"\nZenith Sparse Attention Final Results:")
        print(f"Average Computational Savings: {avg_savings:.2%}")
        print(f"Final Sparsity Ratio: {final_stats['sparsity_ratio']:.3f}")
        print(f"Total Attention Blocks Pruned: {final_stats['attention_blocks_pruned']}")
        
        print("All training phases finished!")
        torch.save(self.model.state_dict(), os.path.join(self.config.CHECKPOINT_DIR, "zenith_protocol_final.pth"))
        torch.save(self.zenith_attention.state_dict(), os.path.join(self.config.CHECKPOINT_DIR, "zenith_attention_final.pth"))
        print(f"Final model and sparse attention saved to {self.config.CHECKPOINT_DIR}")

if __name__ == '__main__':
    orchestrator = ZenithTrainingOrchestrator()
    orchestrator.run_all_phases()