"""
Federated Trainer - Distributed training across multiple devices/users
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict
import hashlib
import time

class FederatedAggregationMethod(Enum):
    FED_AVG = "fed_avg"
    FED_PROX = "fed_prox"
    FED_YOGI = "fed_yogi"
    FED_ADAM = "fed_adam"
    WEIGHTED_AVG = "weighted_avg"

class ClientSelectionStrategy(Enum):
    RANDOM = "random"
    STRATIFIED = "stratified"
    CAPABILITY_BASED = "capability_based"
    PERFORMANCE_BASED = "performance_based"

@dataclass
class FederatedConfig:
    aggregation_method: FederatedAggregationMethod
    client_selection: ClientSelectionStrategy
    clients_per_round: int
    total_rounds: int
    local_epochs: int
    learning_rate: float = 0.001
    privacy_budget: float = 0.1  # For differential privacy
    compression_ratio: float = 0.3  # For communication efficiency

class FederatedTrainer:
    def __init__(self, global_model: nn.Module):
        self.global_model = global_model
        self.client_models = {}
        self.client_metadata = {}
        self.training_history = []
        self.aggregation_weights = {}
        
    def setup_federated_training(self, config: FederatedConfig) -> Dict[str, Any]:
        """Setup federated training environment"""
        
        print(f"🌐 Setting up federated training...")
        print(f"   Method: {config.aggregation_method.value}")
        print(f"   Clients per round: {config.clients_per_round}")
        print(f"   Total rounds: {config.total_rounds}")
        
        self.config = config
        self._initialize_client_registry()
        
        return {
            'status': 'ready',
            'global_model_parameters': sum(p.numel() for p in self.global_model.parameters()),
            'expected_communication_savings': self._estimate_communication_savings(),
            'privacy_guarantees': self._calculate_privacy_guarantees()
        }
    
    def register_client(self, client_id: str, device_capabilities: Dict[str, Any],
                       data_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Register a client for federated training"""
        
        print(f"📱 Registering client {client_id}...")
        
        # Create client-specific model
        client_model = self._create_client_model(client_id)
        
        # Store client metadata
        self.client_metadata[client_id] = {
            'capabilities': device_capabilities,
            'data_stats': data_characteristics,
            'performance_history': [],
            'last_active': time.time(),
            'trust_score': 1.0,  # Start with full trust
            'registration_time': time.time()
        }
        
        self.client_models[client_id] = client_model
        
        # Calculate client weight for aggregation
        self.aggregation_weights[client_id] = self._calculate_client_weight(
            client_id, device_capabilities, data_characteristics
        )
        
        return {
            'client_id': client_id,
            'model_size': sum(p.numel() for p in client_model.parameters()),
            'assigned_weight': self.aggregation_weights[client_id],
            'status': 'active'
        }
    
    def run_training_round(self, round_number: int) -> Dict[str, Any]:
        """Run one round of federated training"""
        
        print(f"🔄 Running federated training round {round_number}...")
        
        # Select clients for this round
        selected_clients = self._select_clients_for_round()
        print(f"   Selected {len(selected_clients)} clients: {selected_clients}")
        
        # Local training on selected clients
        client_updates = {}
        client_metrics = {}
        
        for client_id in selected_clients:
            print(f"   Training client {client_id}...")
            
            client_update, metrics = self._train_client_locally(client_id)
            client_updates[client_id] = client_update
            client_metrics[client_id] = metrics
            
            # Update client metadata
            self._update_client_metadata(client_id, metrics)
        
        # Aggregate updates
        global_update = self._aggregate_updates(client_updates)
        
        # Apply global update
        self._apply_global_update(global_update)
        
        # Calculate round statistics
        round_stats = self._calculate_round_stats(client_metrics, round_number)
        
        print(f"✅ Round {round_number} completed")
        print(f"   Average client accuracy: {round_stats['average_accuracy']:.3f}")
        print(f"   Communication cost: {round_stats['communication_cost_mb']:.2f}MB")
        
        return round_stats
    
    def _select_clients_for_round(self) -> List[str]:
        """Select clients for the current training round"""
        
        active_clients = list(self.client_models.keys())
        
        if self.config.client_selection == ClientSelectionStrategy.RANDOM:
            # Random selection
            selected = np.random.choice(
                active_clients,
                size=min(self.config.clients_per_round, len(active_clients)),
                replace=False
            )
            return selected.tolist()
        
        elif self.config.client_selection == ClientSelectionStrategy.CAPABILITY_BASED:
            # Select based on device capabilities
            client_scores = []
            for client_id in active_clients:
                capabilities = self.client_metadata[client_id]['capabilities']
                score = self._calculate_capability_score(capabilities)
                client_scores.append((client_id, score))
            
            # Sort by capability score and select top
            client_scores.sort(key=lambda x: x[1], reverse=True)
            selected = [client_id for client_id, _ in client_scores[:self.config.clients_per_round]]
            return selected
        
        elif self.config.client_selection == ClientSelectionStrategy.PERFORMANCE_BASED:
            # Select based on past performance
            client_scores = []
            for client_id in active_clients:
                history = self.client_metadata[client_id]['performance_history']
                if history:
                    avg_performance = np.mean([h['accuracy'] for h in history[-3:]])  # Last 3 rounds
                else:
                    avg_performance = 0.5  # Default
                client_scores.append((client_id, avg_performance))
            
            client_scores.sort(key=lambda x: x[1], reverse=True)
            selected = [client_id for client_id, _ in client_scores[:self.config.clients_per_round]]
            return selected
        
        else:
            # Default to random
            return np.random.choice(active_clients, 
                                 size=min(self.config.clients_per_round, len(active_clients)),
                                 replace=False).tolist()
    
    def _train_client_locally(self, client_id: str) -> tuple:
        """Train model locally on client device"""
        
        client_model = self.client_models[client_id]
        client_metadata = self.client_metadata[client_id]
        
        # Sync with global model
        self._sync_client_with_global(client_id)
        
        # Simulate local training (would use client's actual data)
        local_optimizer = torch.optim.Adam(client_model.parameters(), 
                                         lr=self.config.learning_rate)
        
        training_metrics = {
            'loss_history': [],
            'accuracy_history': [],
            'training_time': 0,
            'samples_processed': 0
        }
        
        # Local training loop
        start_time = time.time()
        for epoch in range(self.config.local_epochs):
            # Simulate training on client data
            batch_loss = self._simulate_client_training(client_model, local_optimizer, 
                                                      client_metadata)
            training_metrics['loss_history'].append(batch_loss)
            
            # Simulate accuracy calculation
            accuracy = max(0.5, 1 - batch_loss)  # Simplified
            training_metrics['accuracy_history'].append(accuracy)
        
        training_metrics['training_time'] = time.time() - start_time
        training_metrics['samples_processed'] = 1000  # Simulated
        
        # Calculate model update (difference from global model)
        client_update = self._calculate_client_update(client_id)
        
        # Apply compression to reduce communication
        compressed_update = self._compress_client_update(client_update)
        
        return compressed_update, training_metrics
    
    def _aggregate_updates(self, client_updates: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Aggregate client updates using specified method"""
        
        print(f"   Aggregating {len(client_updates)} client updates...")
        
        if self.config.aggregation_method == FederatedAggregationMethod.FED_AVG:
            return self._fed_avg_aggregation(client_updates)
        elif self.config.aggregation_method == FederatedAggregationMethod.WEIGHTED_AVG:
            return self._weighted_avg_aggregation(client_updates)
        elif self.config.aggregation_method == FederatedAggregationMethod.FED_PROX:
            return self._fed_prox_aggregation(client_updates)
        else:
            return self._fed_avg_aggregation(client_updates)  # Default
    
    def _fed_avg_aggregation(self, client_updates: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Federated Averaging aggregation"""
        
        aggregated_update = {}
        total_weight = sum(self.aggregation_weights[client_id] 
                         for client_id in client_updates.keys())
        
        # Initialize aggregated update
        first_client = next(iter(client_updates.keys()))
        for param_name in client_updates[first_client].keys():
            aggregated_update[param_name] = torch.zeros_like(
                client_updates[first_client][param_name]
            )
        
        # Weighted average of updates
        for client_id, update in client_updates.items():
            weight = self.aggregation_weights[client_id] / total_weight
            
            for param_name, param_update in update.items():
                aggregated_update[param_name] += weight * param_update
        
        return aggregated_update
    
    def _weighted_avg_aggregation(self, client_updates: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Weighted average based on client data size"""
        
        aggregated_update = {}
        total_samples = sum(self.client_metadata[client_id]['data_stats'].get('samples', 1)
                          for client_id in client_updates.keys())
        
        # Initialize aggregated update
        first_client = next(iter(client_updates.keys()))
        for param_name in client_updates[first_client].keys():
            aggregated_update[param_name] = torch.zeros_like(
                client_updates[first_client][param_name]
            )
        
        # Sample-weighted average
        for client_id, update in client_updates.items():
            client_samples = self.client_metadata[client_id]['data_stats'].get('samples', 1)
            weight = client_samples / total_samples
            
            for param_name, param_update in update.items():
                aggregated_update[param_name] += weight * param_update
        
        return aggregated_update
    
    def _fed_prox_aggregation(self, client_updates: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """FedProx aggregation with proximal term"""
        
        # Regular FedAvg with additional proximal term would be applied during local training
        # For simplicity, we'll use weighted average here
        return self._weighted_avg_aggregation(client_updates)
    
    def _apply_global_update(self, global_update: Dict[str, torch.Tensor]):
        """Apply aggregated update to global model"""
        
        with torch.no_grad():
            for param_name, param_update in global_update.items():
                # Find parameter in global model
                for global_param_name, global_param in self.global_model.named_parameters():
                    if global_param_name == param_name:
                        global_param.data += param_update
                        break
    
    def adaptive_client_selection(self, performance_threshold: float = 0.7):
        """Dynamically adjust client selection based on performance"""
        
        def selection_adapter(current_round_stats):
            avg_accuracy = current_round_stats['average_accuracy']
            
            if avg_accuracy < performance_threshold:
                # Switch to performance-based selection
                self.config.client_selection = ClientSelectionStrategy.PERFORMANCE_BASED
                print("   🔄 Switching to performance-based client selection")
            else:
                # Use capability-based for efficiency
                self.config.client_selection = ClientSelectionStrategy.CAPABILITY_BASED
        
        return selection_adapter
    
    def privacy_preserving_aggregation(self, noise_scale: float = 0.01):
        """Add differential privacy noise to aggregation"""
        
        def privacy_wrapper(aggregation_function):
            def noisy_aggregation(client_updates):
                clean_aggregation = aggregation_function(client_updates)
                
                # Add Gaussian noise for differential privacy
                noisy_aggregation = {}
                for param_name, param_update in clean_aggregation.items():
                    noise = torch.randn_like(param_update) * noise_scale
                    noisy_aggregation[param_name] = param_update + noise
                
                return noisy_aggregation
            return noisy_aggregation
        
        # Wrap the current aggregation method
        self._aggregate_updates = privacy_wrapper(self._aggregate_updates)
        print(f"   🔒 Enabled differential privacy (noise scale: {noise_scale})")
    
    def get_federated_stats(self) -> Dict[str, Any]:
        """Get federated training statistics"""
        
        total_clients = len(self.client_models)
        active_clients = [cid for cid, metadata in self.client_metadata.items()
                         if time.time() - metadata['last_active'] < 3600]  # Active in last hour
        
        communication_cost = sum(round_stats.get('communication_cost_mb', 0)
                               for round_stats in self.training_history)
        
        return {
            'total_registered_clients': total_clients,
            'currently_active_clients': len(active_clients),
            'total_training_rounds': len(self.training_history),
            'total_communication_cost_mb': communication_cost,
            'average_round_accuracy': np.mean([r['average_accuracy'] 
                                             for r in self.training_history[-10:]]),
            'client_trust_distribution': self._get_trust_distribution(),
            'estimated_co2_savings': self._estimate_co2_savings(communication_cost)
        }
    
    # Helper methods
    def _initialize_client_registry(self):
        """Initialize client registry"""
        self.client_models = {}
        self.client_metadata = {}
        self.aggregation_weights = {}
    
    def _create_client_model(self, client_id: str) -> nn.Module:
        """Create a client-specific model instance"""
        # Return a copy of the global model
        return type(self.global_model)()
    
    def _calculate_client_weight(self, client_id: str, capabilities: Dict[str, Any],
                               data_stats: Dict[str, Any]) -> float:
        """Calculate client weight for aggregation"""
        
        base_weight = 1.0
        
        # Adjust based on data quality
        data_quality = data_stats.get('quality_score', 0.5)
        base_weight *= data_quality
        
        # Adjust based on device capabilities
        capability_score = self._calculate_capability_score(capabilities)
        base_weight *= capability_score
        
        # Adjust based on data quantity
        data_quantity = min(data_stats.get('samples', 1) / 1000, 2.0)  # Cap at 2x
        base_weight *= data_quantity
        
        return base_weight
    
    def _calculate_capability_score(self, capabilities: Dict[str, Any]) -> float:
        """Calculate device capability score"""
        
        score = 0.5  # Base score
        
        # Memory score
        memory_gb = capabilities.get('memory_gb', 2)
        score += min(memory_gb / 16, 0.3)  # Up to 30% for memory
        
        # Compute score
        compute_tflops = capabilities.get('compute_tflops', 1)
        score += min(compute_tflops / 10, 0.2)  # Up to 20% for compute
        
        return min(score, 1.0)
    
    def _sync_client_with_global(self, client_id: str):
        """Sync client model with global model"""
        client_model = self.client_models[client_id]
        client_model.load_state_dict(self.global_model.state_dict())
    
    def _calculate_client_update(self, client_id: str) -> Dict[str, torch.Tensor]:
        """Calculate difference between client and global model"""
        
        client_model = self.client_models[client_id]
        update = {}
        
        for (client_param_name, client_param), (global_param_name, global_param) in \
            zip(client_model.named_parameters(), self.global_model.named_parameters()):
            
            if client_param_name == global_param_name:
                update[client_param_name] = client_param.data - global_param.data
        
        return update
    
    def _compress_client_update(self, update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compress client update to reduce communication"""
        
        if self.config.compression_ratio >= 1.0:
            return update  # No compression
        
        compressed_update = {}
        
        for param_name, param_update in update.items():
            # Simple top-k compression
            param_flat = param_update.flatten()
            k = int(param_flat.numel() * self.config.compression_ratio)
            
            if k > 0:
                # Keep only top-k values by magnitude
                values, indices = torch.topk(param_flat.abs(), k)
                compressed_param = torch.zeros_like(param_flat)
                compressed_param[indices] = param_flat[indices]
                compressed_update[param_name] = compressed_param.reshape(param_update.shape)
            else:
                compressed_update[param_name] = param_update
        
        return compressed_update
    
    def _calculate_round_stats(self, client_metrics: Dict[str, Any], round_number: int) -> Dict[str, Any]:
        """Calculate statistics for the training round"""
        
        avg_accuracy = np.mean([metrics['accuracy_history'][-1] 
                              for metrics in client_metrics.values()])
        
        avg_loss = np.mean([metrics['loss_history'][-1] 
                          for metrics in client_metrics.values()])
        
        # Estimate communication cost
        communication_cost = self._estimate_communication_cost(client_metrics)
        
        round_stats = {
            'round_number': round_number,
            'clients_participated': len(client_metrics),
            'average_accuracy': avg_accuracy,
            'average_loss': avg_loss,
            'communication_cost_mb': communication_cost,
            'timestamp': time.time()
        }
        
        self.training_history.append(round_stats)
        return round_stats
    
    def _estimate_communication_cost(self, client_metrics: Dict[str, Any]) -> float:
        """Estimate communication cost for the round"""
        
        # Simplified estimation: model size * compression ratio * number of clients
        model_size_mb = sum(p.numel() * 4 for p in self.global_model.parameters()) / (1024**2)  # MB
        compressed_size = model_size_mb * self.config.compression_ratio
        total_communication = compressed_size * len(client_metrics) * 2  # Upload + download
        
        return total_communication
    
    def _update_client_metadata(self, client_id: str, metrics: Dict[str, Any]):
        """Update client metadata with training results"""
        
        metadata = self.client_metadata[client_id]
        metadata['last_active'] = time.time()
        metadata['performance_history'].append({
            'accuracy': metrics['accuracy_history'][-1],
            'loss': metrics['loss_history'][-1],
            'training_time': metrics['training_time']
        })
        
        # Update trust score based on performance consistency
        self._update_client_trust_score(client_id)
    
    def _update_client_trust_score(self, client_id: str):
        """Update client trust score based on performance"""
        
        history = self.client_metadata[client_id]['performance_history']
        if len(history) < 2:
            return
        
        # Calculate performance consistency
        recent_accuracies = [h['accuracy'] for h in history[-3:]]
        consistency = 1.0 - np.std(recent_accuracies)  # Higher std = lower consistency
        
        # Update trust score
        current_trust = self.client_metadata[client_id]['trust_score']
        new_trust = 0.9 * current_trust + 0.1 * consistency
        self.client_metadata[client_id]['trust_score'] = new_trust
    
    def _get_trust_distribution(self) -> Dict[str, int]:
        """Get distribution of client trust scores"""
        
        trust_ranges = {
            'high_trust': 0,    # 0.8-1.0
            'medium_trust': 0,  # 0.5-0.8
            'low_trust': 0,     # 0.0-0.5
        }
        
        for metadata in self.client_metadata.values():
            trust_score = metadata['trust_score']
            if trust_score >= 0.8:
                trust_ranges['high_trust'] += 1
            elif trust_score >= 0.5:
                trust_ranges['medium_trust'] += 1
            else:
                trust_ranges['low_trust'] += 1
        
        return trust_ranges
    
    def _estimate_co2_savings(self, communication_cost_mb: float) -> float:
        """Estimate CO2 savings from federated vs centralized training"""
        
        # Simplified estimation
        # Centralized training would require data transfer to central server
        # Federated only transfers model updates
        estimated_data_transfer_savings = communication_cost_mb * 10  # Arbitrary multiplier
        co2_savings_kg = estimated_data_transfer_savings * 0.0001  # Rough estimate
        
        return co2_savings_kg
    
    def _estimate_communication_savings(self) -> float:
        """Estimate communication savings from federated learning"""
        
        # Compared to centralized training where all data is transferred
        estimated_centralized_cost = 1000  # MB (example)
        estimated_federated_cost = 50     # MB per round (example)
        
        savings = (estimated_centralized_cost - estimated_federated_cost) / estimated_centralized_cost
        return max(savings, 0)
    
    def _calculate_privacy_guarantees(self) -> Dict[str, float]:
        """Calculate privacy guarantees"""
        
        return {
            'epsilon': 1.0 / (1 + self.config.privacy_budget),
            'delta': 1e-5,
            'privacy_level': min(1.0, self.config.privacy_budget * 10)
        }
    
    def _simulate_client_training(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                                metadata: Dict[str, Any]) -> float:
        """Simulate training on client data"""
        
        # This would be replaced with actual client data training
        # For simulation, return a random loss that improves over time
        base_loss = 0.5
        capability_factor = self._calculate_capability_score(metadata['capabilities'])
        data_quality = metadata['data_stats'].get('quality_score', 0.5)
        
        # Simulated loss improvement
        simulated_loss = base_loss * (1 - capability_factor * 0.3) * (1 - data_quality * 0.2)
        simulated_loss *= np.random.uniform(0.8, 1.2)  # Add some randomness
        
        return max(simulated_loss, 0.1)
