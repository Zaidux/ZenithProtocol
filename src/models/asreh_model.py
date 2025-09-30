# /src/models/asreh_model.py

"""
Enhanced ASREH Model with Causal Reasoning Integration
======================================================
Now provides better integration with the ASREH algorithm and CKG-based reasoning.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from ..modules.mixture_of_experts import MixtureOfExperts
from copy import deepcopy

from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph
from ..web_access.web_access import WebAccess
import asreh_model_cpp
import moe_router_cpp

class ASREHModel(nn.Module):
    """
    Adaptive Self-Regulating Explainable Hybrid Model
    Now enhanced with better algorithm integration and causal reasoning support.
    """
    def __init__(self,
                 in_channels: int = 1,
                 num_experts: int = 4,
                 hct_dim: int = 64,
                 ckg: ConceptualKnowledgeGraph = None,
                 web_access: WebAccess = None):

        super(ASREHModel, self).__init__()
        self.hct_dim = hct_dim
        self.in_channels = in_channels
        self.num_experts = num_experts

        self.ckg = ckg or ConceptualKnowledgeGraph()
        self.web_access = web_access or WebAccess(self.ckg)

        # C++ components for performance
        self.cpp_asreh_model = asreh_model_cpp.ASREHModel(
            in_channels=in_channels,
            hct_dim=hct_dim,
            num_experts=num_experts
        )
        self.cpp_moe_router = moe_router_cpp.ConceptualAwareRouter(
            input_dim=hct_dim,
            num_experts=num_experts,
            top_k=2
        )

        # Enhanced encoder architecture
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, hct_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hct_dim),
        )

        # Enhanced conceptual attention layer
        self.conceptual_attention_layer = nn.Sequential(
            nn.Linear(hct_dim, hct_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hct_dim * 2, hct_dim),
            nn.ReLU(),
            nn.LayerNorm(hct_dim)
        )

        # State prediction decoder
        self.state_decoder = nn.Sequential(
            nn.Linear(hct_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64 * 64),
            nn.Sigmoid()  # For binary board prediction
        )

        # Conceptual feature projector
        self.conceptual_projector = nn.Sequential(
            nn.Linear(hct_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),  # Standardized conceptual feature size
            nn.Tanh()
        )

        # Domain adaptation parameters
        self.domain_adaptation = nn.ParameterDict({
            'tetris': nn.Parameter(torch.randn(hct_dim)),
            'chess': nn.Parameter(torch.randn(hct_dim)),
            'general': nn.Parameter(torch.randn(hct_dim))
        })

        # Performance monitoring
        self.performance_metrics = {
            'inference_count': 0,
            'avg_confidence': 0.0,
            'domain_usage': {'tetris': 0, 'chess': 0, 'other': 0}
        }

    def forward(self, state: torch.Tensor, conceptual_features: torch.Tensor, 
                domain: str, return_intermediate: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Enhanced forward pass with better integration for the ASREH algorithm.
        
        Args:
            state: Input state tensor
            conceptual_features: Pre-computed conceptual features
            domain: Problem domain for domain adaptation
            return_intermediate: Whether to return intermediate representations
            
        Returns:
            predicted_state: Decoded state prediction
            fused_representation: Combined representation for decision making
            confidence: Prediction confidence score
        """
        # Encode input state
        encoded_state = self.shared_encoder(state)
        
        # Flatten and process through conceptual attention
        batch_size, channels, height, width = encoded_state.shape
        encoded_flat = encoded_state.view(batch_size, -1)
        
        # Apply conceptual attention
        attended_features = self.conceptual_attention_layer(encoded_flat)
        
        # Fuse with external conceptual features
        fused_representation = self._fuse_representations(attended_features, conceptual_features, domain)
        
        # Decode to predicted state
        predicted_state = self.state_decoder(fused_representation)
        
        # Project to standardized conceptual features
        standardized_features = self.conceptual_projector(fused_representation)
        
        # Calculate confidence
        confidence = self._calculate_confidence(fused_representation, predicted_state)
        
        # Update performance metrics
        self._update_performance_metrics(domain, confidence.item())
        
        if return_intermediate:
            return predicted_state, fused_representation, standardized_features, confidence
        else:
            # Reshape based on domain
            if domain == 'tetris':
                return predicted_state.view(batch_size, 1, 20, 10), fused_representation, confidence
            elif domain == 'chess':
                return predicted_state, fused_representation, confidence
            else:
                return predicted_state, fused_representation, confidence

    def _fuse_representations(self, attended_features: torch.Tensor, 
                            conceptual_features: torch.Tensor, domain: str) -> torch.Tensor:
        """
        Fuse encoded features with external conceptual features.
        """
        # Use C++ implementation for performance
        fused_np = self.cpp_asreh_model.forward(
            attended_features.detach().cpu().numpy(),
            conceptual_features.detach().cpu().numpy()
        )
        fused_tensor = torch.from_numpy(fused_np).to(attended_features.device)
        
        # Apply domain-specific adaptation
        domain_factor = self.domain_adaptation.get(domain, self.domain_adaptation['general'])
        fused_tensor = fused_tensor * domain_factor.unsqueeze(0)
        
        return fused_tensor

    def _calculate_confidence(self, fused_representation: torch.Tensor, 
                            predicted_state: torch.Tensor) -> torch.Tensor:
        """
        Calculate prediction confidence based on representation consistency.
        """
        # Measure variance in fused representation
        rep_variance = torch.var(fused_representation, dim=1)
        rep_confidence = torch.exp(-rep_variance)
        
        # Measure prediction certainty (for binary predictions)
        if predicted_state.shape[1] == 1:  # Binary prediction
            pred_confidence = 1.0 - torch.abs(predicted_state - 0.5) * 2.0
            pred_confidence = torch.mean(pred_confidence)
        else:
            pred_confidence = torch.tensor(0.8)  # Default confidence
        
        # Combined confidence
        confidence = (rep_confidence + pred_confidence) / 2.0
        return confidence

    def _update_performance_metrics(self, domain: str, confidence: float):
        """Update internal performance tracking metrics."""
        self.performance_metrics['inference_count'] += 1
        self.performance_metrics['avg_confidence'] = (
            self.performance_metrics['avg_confidence'] * 0.9 + confidence * 0.1
        )
        
        if domain in self.performance_metrics['domain_usage']:
            self.performance_metrics['domain_usage'][domain] += 1
        else:
            self.performance_metrics['domain_usage']['other'] += 1

    def get_conceptual_features(self, state: torch.Tensor, domain: str) -> torch.Tensor:
        """
        Extract conceptual features from state for CKG integration.
        
        Args:
            state: Input state tensor
            domain: Problem domain
            
        Returns:
            Standardized conceptual features for CKG processing
        """
        with torch.no_grad():
            encoded_state = self.shared_encoder(state)
            batch_size, channels, height, width = encoded_state.shape
            encoded_flat = encoded_state.view(batch_size, -1)
            attended_features = self.conceptual_attention_layer(encoded_flat)
            standardized_features = self.conceptual_projector(attended_features)
            
            # Apply domain normalization
            if domain == 'tetris':
                # Normalize for tetris features: [lines_cleared, gaps, max_height, board_fullness]
                standardized_features = standardized_features[:, :4]  # First 4 features
                standardized_features = torch.sigmoid(standardized_features)  # 0-1 range
            elif domain == 'chess':
                # Normalize for chess features
                standardized_features = standardized_features[:, :4]  # First 4 features
                standardized_features = torch.tanh(standardized_features)  # -1 to 1 range
            
            return standardized_features

    def is_struggling(self) -> bool:
        """
        Check if the model is performing poorly based on internal metrics.
        """
        if self.performance_metrics['inference_count'] < 10:
            return False  # Not enough data
        
        return self.performance_metrics['avg_confidence'] < 0.6

    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report."""
        return {
            'total_inferences': self.performance_metrics['inference_count'],
            'average_confidence': self.performance_metrics['avg_confidence'],
            'domain_usage': self.performance_metrics['domain_usage'],
            'is_struggling': self.is_struggling(),
            'parameter_count': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

    def route_to_experts(self, fused_representation: torch.Tensor, domain: str) -> torch.Tensor:
        """
        Route representation to appropriate experts using MoE.
        """
        conceptual_context = moe_router_cpp.ConceptualContext()
        conceptual_context.context_map = {'topic': [domain]}
        
        top_k_indices_np = self.cpp_moe_router.route(
            fused_representation.detach().cpu().numpy(),
            conceptual_context
        )
        top_k_indices = torch.from_numpy(top_k_indices_np).long()
        
        # For now, return original representation - expert processing would happen here
        return fused_representation

    def predict_next_state(self, current_state: torch.Tensor, action: int, 
                          domain: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next state given current state and action.
        """
        # Get conceptual features of current state
        current_features = self.get_conceptual_features(current_state, domain)
        
        # Encode action information (simplified - would be more sophisticated)
        action_tensor = torch.tensor([action], dtype=torch.float32).to(current_state.device)
        action_features = action_tensor.unsqueeze(0).unsqueeze(0).expand(current_features.shape[0], -1)
        
        # Combine with current features
        combined_features = torch.cat([current_features, action_features], dim=1)
        
        # Predict next state
        with torch.no_grad():
            predicted_state, _, confidence = self.forward(
                current_state, combined_features, domain
            )
        
        return predicted_state, confidence

    def adapt_to_domain(self, domain: str, adaptation_strength: float = 0.1):
        """
        Adapt model parameters for better performance in specific domain.
        """
        # This would implement domain-specific adaptation
        # For now, just update the domain adaptation parameter
        if domain in self.domain_adaptation:
            with torch.no_grad():
                self.domain_adaptation[domain].data += adaptation_strength * torch.randn_like(self.domain_adaptation[domain])

    def get_state_dict(self) -> Dict:
        """Get model state dict including performance metrics."""
        state_dict = super().state_dict()
        state_dict['performance_metrics'] = self.performance_metrics
        state_dict['domain_adaptation'] = {k: v.data for k, v in self.domain_adaptation.items()}
        return state_dict

    def set_state_dict(self, state_dict: Dict):
        """Set model state dict including performance metrics."""
        if 'performance_metrics' in state_dict:
            self.performance_metrics = state_dict.pop('performance_metrics')
        if 'domain_adaptation' in state_dict:
            for k, v in state_dict.pop('domain_adaptation').items():
                if k in self.domain_adaptation:
                    self.domain_adaptation[k].data.copy_(v)
        
        self.load_state_dict(state_dict)

    def get_fast_adaptable_model(self):
        """Create a lightweight copy for rapid adaptation."""
        return deepcopy(self)

    def explain_prediction(self, state: torch.Tensor, predicted_state: torch.Tensor, 
                          domain: str) -> Dict:
        """
        Generate explanation for a prediction using CKG integration.
        """
        conceptual_features = self.get_conceptual_features(state, domain)
        
        # Get CKG validation of the prediction
        ckg_validation = self.ckg.validate_forecast(conceptual_features, "prediction", domain)
        
        return {
            'conceptual_features': conceptual_features,
            'ckg_validation': ckg_validation,
            'confidence': self._calculate_confidence(
                self.conceptual_attention_layer(self.shared_encoder(state).view(state.shape[0], -1)),
                predicted_state
            ).item(),
            'domain': domain,
            'applied_rules': ckg_validation.get('applied_rules', []),
            'violated_rules': ckg_validation.get('violated_rules', [])
        }

# Add to existing ASREHModel class
class ASREHModelWithSparseAttention(ASREHModel):
    """
    Enhanced ASREH model with conceptual sparse attention.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Replace standard attention with sparse attention
        self.sparse_attention = ConceptualSparseAttention(
            dim=self.hct_dim,
            num_heads=8,
            ckg=self.ckg,
            sparsity_ratio=0.15
        )
        
        # Multi-modal sparse attention
        self.multi_modal_attention = MultiModalSparseAttention(
            dim=self.hct_dim,
            num_heads=8,
            ckg=self.ckg
        )
    
    def forward(self, state: torch.Tensor, conceptual_features: torch.Tensor, 
                domain: str, return_intermediate: bool = False,
                use_sparse_attention: bool = True):
        
        # Encode input state
        encoded_state = self.shared_encoder(state)
        batch_size, channels, height, width = encoded_state.shape
        encoded_flat = encoded_state.view(batch_size, -1)
        
        # Use sparse attention if enabled and sequence is long enough
        if use_sparse_attention and encoded_flat.shape[1] > 100:
            # Reshape for attention (add sequence dimension)
            encoded_reshaped = encoded_flat.unsqueeze(1)  # [B, 1, D]
            
            # Apply sparse attention with domain context
            context = {'domain': domain, 'sequence_type': 'encoded_state'}
            attended_features = self.sparse_attention(
                encoded_reshaped, context=context
            ).squeeze(1)
        else:
            # Fallback to standard processing
            attended_features = self.conceptual_attention_layer(encoded_flat)
        
        # Rest of the forward pass remains the same
        fused_representation = self._fuse_representations(
            attended_features, conceptual_features, domain
        )
        predicted_state = self.state_decoder(fused_representation)
        standardized_features = self.conceptual_projector(fused_representation)
        confidence = self._calculate_confidence(fused_representation, predicted_state)
        
        self._update_performance_metrics(domain, confidence.item())
        
        if return_intermediate:
            return predicted_state, fused_representation, standardized_features, confidence
        else:
            # Handle different output shapes based on domain
            if domain == 'tetris':
                return predicted_state.view(batch_size, 1, 20, 10), fused_representation, confidence
            elif domain == 'chess':
                return predicted_state, fused_representation, confidence
            else:
                return predicted_state, fused_representation, confidence
    
    def get_attention_performance(self) -> Dict:
        """Get sparse attention performance metrics."""
        base_performance = self.get_performance_report()
        attention_performance = self.sparse_attention.get_performance_report()
        
        return {**base_performance, **attention_performance}

# Add to your existing ASREHModel class
class ASREHModelWithCKGSparseAttention(ASREHModel):
    """
    ASREH model with full CKG-guided sparse attention integration.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Replace with CKG-guided sparse attention
        self.ckg_sparse_attention = CKGSparseAttention(
            dim=self.hct_dim,
            num_heads=8,
            ckg=self.ckg,
            sparsity_ratio=0.15
        )
        
        # Multi-modal CKG attention
        self.multi_modal_ckg_attention = MultiModalCKGAttention(
            dim=self.hct_dim,
            num_heads=8,
            ckg=self.ckg,
            modalities=['text', 'visual', 'audio']
        )
        
        # CKG integration controller
        self.ckg_integration_controller = nn.Sequential(
            nn.Linear(self.hct_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # [use_ckg, sparsity_ratio, confidence_threshold]
            nn.Sigmoid()
        )
    
    def forward(self, state: torch.Tensor, conceptual_features: torch.Tensor,
                domain: str, return_intermediate: bool = False,
                multimodal_inputs: Dict = None):
        
        # Determine CKG integration level
        integration_controls = self.ckg_integration_controller(
            conceptual_features.mean(dim=1) if conceptual_features is not None 
            else torch.zeros(1, self.hct_dim, device=state.device)
        )
        
        use_ckg_guidance = integration_controls[0] > 0.5
        dynamic_sparsity = integration_controls[1] * 0.3  # 0-30% sparsity
        
        # Handle multi-modal inputs
        if multimodal_inputs and len(multimodal_inputs) > 1:
            context = {'domain': domain, 'multimodal': True}
            fused_representation = self.multi_modal_ckg_attention(
                multimodal_inputs, context=context
            )
        else:
            # Single modality processing with CKG guidance
            encoded_state = self.shared_encoder(state)
            batch_size, channels, height, width = encoded_state.shape
            encoded_flat = encoded_state.view(batch_size, -1)
            
            # Apply CKG-guided sparse attention
            context = {'domain': domain, 'sequence_type': 'encoded_state'}
            fused_representation, _, guidance_info = self.ckg_sparse_attention(
                encoded_flat.unsqueeze(1) if encoded_flat.dim() == 2 else encoded_flat,
                context=context,
                return_attention_weights=True,
                use_ckg_guidance=use_ckg_guidance
            )
            
            if encoded_flat.dim() == 2:
                fused_representation = fused_representation.squeeze(1)
        
        # Continue with standard ASREH processing
        predicted_state = self.state_decoder(fused_representation)
        standardized_features = self.conceptual_projector(fused_representation)
        confidence = self._calculate_confidence(fused_representation, predicted_state)
        
        self._update_performance_metrics(domain, confidence.item())
        
        if return_intermediate:
            return predicted_state, fused_representation, standardized_features, confidence
        else:
            # Domain-specific output shaping
            if domain == 'tetris':
                return predicted_state.view(batch_size, 1, 20, 10), fused_representation, confidence
            else:
                return predicted_state, fused_representation, confidence
    
    def get_ckg_integration_report(self) -> Dict:
        """Get comprehensive CKG integration report."""
        base_report = self.get_performance_report()
        attention_report = self.ckg_sparse_attention.get_ckg_performance_report()
        
        if hasattr(self, 'multi_modal_ckg_attention'):
            multimodal_report = self.multi_modal_ckg_attention.get_cross_modal_report()
        else:
            multimodal_report = {}
        
        return {
            **base_report,
            'ckg_attention': attention_report,
            'multimodal_integration': multimodal_report,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'sparse_parameters': sum(p.numel() for p in self.ckg_sparse_attention.parameters())
        }

# Add to your ASREHModel class
class ASREHModelWithSelfEvolvingAttention(ASREHModel):
    """
    ASREH model with full self-evolving sparse attention capabilities.
    The pinnacle of Zenith's adaptive attention optimization.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Ultimate attention mechanism
        self.self_evolving_attention = SelfEvolvingSparseAttention(
            dim=self.hct_dim,
            num_heads=8,
            ckg=self.ckg,
            sparsity_ratio=0.12,  # More aggressive sparsity with evolution
            evolution_interval=50,  # Evolve more frequently
            exploration_rate=0.15
        )
        
        # Evolution monitoring
        self.evolution_tracker = {
            'total_evolutions': 0,
            'performance_trend': [],
            'best_pattern_fitness': 0.0,
            'last_evolution_time': datetime.now()
        }
    
    def forward(self, state: torch.Tensor, conceptual_features: torch.Tensor,
                domain: str, return_intermediate: bool = False,
                enable_evolution: bool = True):
        
        # Encode input state
        encoded_state = self.shared_encoder(state)
        batch_size, channels, height, width = encoded_state.shape
        encoded_flat = encoded_state.view(batch_size, -1)
        
        # Apply self-evolving attention
        context = {
            'domain': domain,
            'sequence_type': 'encoded_state',
            'input_shape': state.shape,
            'conceptual_features_present': conceptual_features is not None
        }
        
        fused_representation, _, evolution_info = self.self_evolving_attention(
            encoded_flat.unsqueeze(1) if encoded_flat.dim() == 2 else encoded_flat,
            context=context,
            return_attention_weights=True,
            enable_evolution=enable_evolution
        )
        
        if encoded_flat.dim() == 2:
            fused_representation = fused_representation.squeeze(1)
        
        # Track evolution events
        if evolution_info.get('evolution_triggered', False):
            self.evolution_tracker['total_evolutions'] += 1
            self.evolution_tracker['last_evolution_time'] = datetime.now()
        
        # Continue with standard processing
        predicted_state = self.state_decoder(fused_representation)
        standardized_features = self.conceptual_projector(fused_representation)
        confidence = self._calculate_confidence(fused_representation, predicted_state)
        
        self._update_performance_metrics(domain, confidence.item())
        
        if return_intermediate:
            return predicted_state, fused_representation, standardized_features, confidence, evolution_info
        else:
            if domain == 'tetris':
                return predicted_state.view(batch_size, 1, 20, 10), fused_representation, confidence
            else:
                return predicted_state, fused_representation, confidence
    
    def get_evolution_insights(self) -> Dict:
        """Get deep insights into the evolutionary process."""
        evolution_report = self.self_evolving_attention.get_evolution_report()
        
        # Calculate performance trends
        recent_performance = self.performance_metrics['avg_confidence']
        self.evolution_tracker['performance_trend'].append(recent_performance)
        if len(self.evolution_tracker['performance_trend']) > 10:
            self.evolution_tracker['performance_trend'].pop(0)
        
        performance_trend = np.mean(self.evolution_tracker['performance_trend'][-5:]) - \
                          np.mean(self.evolution_tracker['performance_trend'][:5]) \
                          if len(self.evolution_tracker['performance_trend']) >= 10 else 0.0
        
        return {
            **evolution_report,
            'total_evolutions': self.evolution_tracker['total_evolutions'],
            'performance_trend': performance_trend,
            'time_since_last_evolution': (
                datetime.now() - self.evolution_tracker['last_evolution_time']
            ).total_seconds(),
            'evolution_effectiveness': 'high' if performance_trend > 0 else 'low'
        }
    
    def trigger_forced_evolution(self, context: Dict):
        """Force an evolution cycle for specific context."""
        print("[Forced Evolution] Triggering manual evolution cycle...")
        
        # Get current sequence length from context or use default
        seq_len = context.get('sequence_length', 256)
        
        # Trigger evolution
        self.self_evolving_attention._trigger_evolution(context, seq_len)
        
        self.evolution_tracker['total_evolutions'] += 1
        self.evolution_tracker['last_evolution_time'] = datetime.now()
        
        return self.get_evolution_insights()
    
    def export_evolution_knowledge(self, filepath: str):
        """Export all evolutionary knowledge for transfer learning."""
        self.self_evolving_attention.export_learned_patterns(filepath)
        
        # Add model-specific evolution data
        evolution_data = {
            'model_evolution_tracker': self.evolution_tracker,
            'performance_metrics': self.performance_metrics,
            'export_timestamp': datetime.now().isoformat(),
            'model_parameters': sum(p.numel() for p in self.parameters()),
            'attention_parameters': sum(p.numel() for p in self.self_evolving_attention.parameters())
        }
        
        print(f"Complete evolution knowledge exported to {filepath}")

# Example usage and testing
if __name__ == '__main__':
    # Create model instance
    ckg = ConceptualKnowledgeGraph()
    model = ASREHModel(ckg=ckg)
    
    # Test with mock data
    mock_state = torch.randn(1, 1, 20, 10)  # Tetris board
    mock_conceptual = torch.randn(1, 4)     # Conceptual features
    
    # Forward pass
    predicted_state, fused_rep, confidence = model(mock_state, mock_conceptual, 'tetris')
    print(f"Prediction confidence: {confidence.item():.3f}")
    
    # Get performance report
    report = model.get_performance_report()
    print(f"Performance: {report}")
    
    # Test conceptual feature extraction
    features = model.get_conceptual_features(mock_state, 'tetris')
    print(f"Conceptual features: {features}")
    
    # Test prediction explanation
    explanation = model.explain_prediction(mock_state, predicted_state, 'tetris')
    print(f"Explanation keys: {list(explanation.keys())}")