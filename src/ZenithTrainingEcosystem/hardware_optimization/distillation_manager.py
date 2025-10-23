"""
Distillation Manager - Knowledge transfer from large models to small ones
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

class DistillationMethod(Enum):
    STANDARD = "standard"
    ATTENTION = "attention"
    HIDDEN_STATES = "hidden_states"
    RELATIONAL = "relational"
    MULTI_HEAD = "multi_head"

@dataclass
class DistillationConfig:
    method: DistillationMethod
    temperature: float = 3.0
    alpha: float = 0.7  # Weight for distillation loss vs task loss
    layer_matching: str = "auto"  # auto, manual, progressive
    attention_transfer: bool = True

class DistillationManager:
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module):
        self.teacher = teacher_model
        self.student = student_model
        self.distillation_stats = {}
        
    def setup_distillation(self, config: DistillationConfig) -> Dict[str, Any]:
        """Setup knowledge distillation process"""
        
        print(f"🎓 Setting up {config.method.value} distillation...")
        print(f"   Temperature: {config.temperature}")
        print(f"   Alpha: {config.alpha}")
        
        self.config = config
        self._setup_distillation_hooks()
        
        # Calculate model sizes
        teacher_params = sum(p.numel() for p in self.teacher.parameters())
        student_params = sum(p.numel() for p in self.student.parameters())
        compression_ratio = teacher_params / student_params
        
        self.distillation_stats = {
            'teacher_parameters': teacher_params,
            'student_parameters': student_params,
            'compression_ratio': compression_ratio,
            'distillation_loss': [],
            'task_loss': [],
            'current_temperature': config.temperature,
            'best_student_accuracy': 0.0
        }
        
        return {
            'status': 'ready',
            'compression_ratio': compression_ratio,
            'estimated_speedup': self._estimate_speedup(compression_ratio),
            'method_details': self._get_method_details(config.method)
        }
    
    def distill_knowledge(self, dataloader, num_epochs: int = 10) -> Dict[str, Any]:
        """Perform knowledge distillation training"""
        
        print(f"🔥 Starting knowledge distillation for {num_epochs} epochs...")
        
        optimizer = torch.optim.Adam(self.student.parameters(), lr=1e-4)
        best_accuracy = 0.0
        
        for epoch in range(num_epochs):
            epoch_losses = self._distillation_epoch(dataloader, optimizer, epoch)
            
            # Update statistics
            self.distillation_stats['distillation_loss'].append(epoch_losses['distillation_loss'])
            self.distillation_stats['task_loss'].append(epoch_losses['task_loss'])
            
            # Evaluate student
            student_accuracy = self._evaluate_student(dataloader)
            best_accuracy = max(best_accuracy, student_accuracy)
            self.distillation_stats['best_student_accuracy'] = best_accuracy
            
            print(f"   Epoch {epoch+1}: "
                  f"Distill Loss: {epoch_losses['distillation_loss']:.4f}, "
                  f"Task Loss: {epoch_losses['task_loss']:.4f}, "
                  f"Accuracy: {student_accuracy:.3f}")
            
            # Adjust temperature gradually
            if epoch % 3 == 0 and self.config.temperature > 1.0:
                self.config.temperature *= 0.9  # Gradually reduce temperature
                self.distillation_stats['current_temperature'] = self.config.temperature
        
        print(f"✅ Distillation completed. Best accuracy: {best_accuracy:.3f}")
        
        return self.get_distillation_results()
    
    def _distillation_epoch(self, dataloader, optimizer, epoch: int) -> Dict[str, float]:
        """Run one epoch of distillation training"""
        
        self.teacher.eval()
        self.student.train()
        
        total_distill_loss = 0.0
        total_task_loss = 0.0
        total_batches = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Get teacher predictions (no gradients)
            with torch.no_grad():
                teacher_outputs = self.teacher(data)
            
            # Get student predictions
            student_outputs = self.student(data)
            
            # Calculate distillation loss
            distill_loss = self._calculate_distillation_loss(teacher_outputs, student_outputs)
            
            # Calculate task loss (if targets available)
            task_loss = F.cross_entropy(student_outputs, target) if target is not None else 0
            
            # Combined loss
            combined_loss = (self.config.alpha * distill_loss + 
                           (1 - self.config.alpha) * task_loss)
            
            combined_loss.backward()
            optimizer.step()
            
            total_distill_loss += distill_loss.item()
            total_task_loss += task_loss.item() if target is not None else 0
            total_batches += 1
            
            if batch_idx % 100 == 0:
                print(f"     Batch {batch_idx}: Loss = {combined_loss.item():.4f}")
        
        return {
            'distillation_loss': total_distill_loss / total_batches,
            'task_loss': total_task_loss / total_batches if total_task_loss > 0 else 0
        }
    
    def _calculate_distillation_loss(self, teacher_outputs, student_outputs) -> torch.Tensor:
        """Calculate distillation loss based on selected method"""
        
        if self.config.method == DistillationMethod.STANDARD:
            return self._standard_distillation_loss(teacher_outputs, student_outputs)
        elif self.config.method == DistillationMethod.ATTENTION:
            return self._attention_distillation_loss(teacher_outputs, student_outputs)
        elif self.config.method == DistillationMethod.HIDDEN_STATES:
            return self._hidden_states_distillation_loss(teacher_outputs, student_outputs)
        elif self.config.method == DistillationMethod.RELATIONAL:
            return self._relational_distillation_loss(teacher_outputs, student_outputs)
        else:
            return self._standard_distillation_loss(teacher_outputs, student_outputs)
    
    def _standard_distillation_loss(self, teacher_outputs, student_outputs) -> torch.Tensor:
        """Standard knowledge distillation loss"""
        
        # Soften teacher outputs with temperature
        teacher_soft = F.softmax(teacher_outputs / self.config.temperature, dim=-1)
        student_soft = F.log_softmax(student_outputs / self.config.temperature, dim=-1)
        
        # KL divergence loss
        distill_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        distill_loss *= self.config.temperature ** 2  # Scale back
        
        return distill_loss
    
    def _attention_distillation_loss(self, teacher_outputs, student_outputs) -> torch.Tensor:
        """Attention-based distillation loss"""
        
        # This would extract attention maps from both models
        # For now, use standard distillation as fallback
        standard_loss = self._standard_distillation_loss(teacher_outputs, student_outputs)
        
        if hasattr(self.teacher, 'attention_maps') and hasattr(self.student, 'attention_maps'):
            attention_loss = 0.0
            for t_attn, s_attn in zip(self.teacher.attention_maps, self.student.attention_maps):
                attention_loss += F.mse_loss(s_attn, t_attn)
            
            return standard_loss + attention_loss * 0.1  # Weight attention loss
        
        return standard_loss
    
    def _hidden_states_distillation_loss(self, teacher_outputs, student_outputs) -> torch.Tensor:
        """Hidden states matching distillation"""
        
        standard_loss = self._standard_distillation_loss(teacher_outputs, student_outputs)
        
        if hasattr(self.teacher, 'hidden_states') and hasattr(self.student, 'hidden_states'):
            hidden_loss = 0.0
            for t_hidden, s_hidden in zip(self.teacher.hidden_states, self.student.hidden_states):
                # Match hidden state distributions
                hidden_loss += F.mse_loss(
                    F.normalize(s_hidden, p=2, dim=-1),
                    F.normalize(t_hidden, p=2, dim=-1)
                )
            
            return standard_loss + hidden_loss * 0.05
        
        return standard_loss
    
    def _relational_distillation_loss(self, teacher_outputs, student_outputs) -> torch.Tensor:
        """Relational knowledge distillation"""
        
        standard_loss = self._standard_distillation_loss(teacher_outputs, student_outputs)
        
        # Calculate relational knowledge (similarity between samples)
        with torch.no_grad():
            teacher_relational = self._calculate_relational_knowledge(teacher_outputs)
        
        student_relational = self._calculate_relational_knowledge(student_outputs)
        
        relational_loss = F.mse_loss(student_relational, teacher_relational)
        
        return standard_loss + relational_loss * 0.1
    
    def _calculate_relational_knowledge(self, outputs: torch.Tensor) -> torch.Tensor:
        """Calculate relational knowledge matrix"""
        
        # Normalize outputs
        outputs_norm = F.normalize(outputs, p=2, dim=-1)
        
        # Calculate similarity matrix
        relational = torch.mm(outputs_norm, outputs_norm.t())
        
        return relational
    
    def progressive_distillation(self, dataloader, stages: int = 3) -> nn.Module:
        """Progressive knowledge distillation through multiple stages"""
        
        print(f"📈 Starting progressive distillation with {stages} stages...")
        
        original_temperature = self.config.temperature
        original_alpha = self.config.alpha
        
        for stage in range(stages):
            print(f"   Stage {stage + 1}/{stages}")
            
            # Adjust parameters for this stage
            self.config.temperature = original_temperature * (0.7 ** stage)
            self.config.alpha = original_alpha * (0.8 ** stage)
            
            print(f"     Temperature: {self.config.temperature:.2f}, Alpha: {self.config.alpha:.2f}")
            
            # Distill for a few epochs
            self.distill_knowledge(dataloader, num_epochs=2)
        
        # Restore original parameters
        self.config.temperature = original_temperature
        self.config.alpha = original_alpha
        
        return self.student
    
    def multi_teacher_distillation(self, teachers: List[nn.Module], 
                                 dataloader, weights: List[float] = None) -> nn.Module:
        """Distill knowledge from multiple teacher models"""
        
        print(f"👥 Multi-teacher distillation with {len(teachers)} teachers...")
        
        if weights is None:
            weights = [1.0 / len(teachers)] * len(teachers)  # Equal weights
        
        original_teacher = self.teacher
        best_accuracy = 0.0
        
        for epoch in range(5):  # Fewer epochs for multi-teacher
            self.student.train()
            total_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(dataloader):
                # Get predictions from all teachers
                teacher_outputs = []
                for teacher in teachers:
                    with torch.no_grad():
                        outputs = teacher(data)
                        teacher_outputs.append(outputs)
                
                # Student prediction
                student_outputs = self.student(data)
                
                # Combined distillation loss
                distill_loss = 0.0
                for teacher_out, weight in zip(teacher_outputs, weights):
                    distill_loss += weight * self._standard_distillation_loss(
                        teacher_out, student_outputs
                    )
                
                # Task loss
                task_loss = F.cross_entropy(student_outputs, target) if target is not None else 0
                
                # Combined loss
                combined_loss = (self.config.alpha * distill_loss + 
                               (1 - self.config.alpha) * task_loss)
                
                combined_loss.backward()
                
                total_loss += combined_loss.item()
                
                if batch_idx % 100 == 0:
                    print(f"     Batch {batch_idx}: Loss = {combined_loss.item():.4f}")
            
            # Evaluate
            accuracy = self._evaluate_student(dataloader)
            best_accuracy = max(best_accuracy, accuracy)
            print(f"   Epoch {epoch+1}: Loss = {total_loss/(batch_idx+1):.4f}, "
                  f"Accuracy = {accuracy:.3f}")
        
        # Restore original teacher
        self.teacher = original_teacher
        
        print(f"✅ Multi-teacher distillation completed. Best accuracy: {best_accuracy:.3f}")
        return self.student
    
    def get_distillation_results(self) -> Dict[str, Any]:
        """Get distillation results and statistics"""
        
        results = self.distillation_stats.copy()
        
        # Calculate knowledge transfer efficiency
        teacher_accuracy = self._evaluate_teacher()  # Would use validation set
        student_accuracy = self.distillation_stats['best_student_accuracy']
        
        knowledge_transfer_efficiency = student_accuracy / teacher_accuracy
        
        results.update({
            'teacher_accuracy': teacher_accuracy,
            'student_accuracy': student_accuracy,
            'knowledge_transfer_efficiency': knowledge_transfer_efficiency,
            'performance_ratio': student_accuracy / teacher_accuracy,
            'estimated_inference_speedup': self._estimate_inference_speedup(),
            'memory_reduction': self._calculate_memory_reduction()
        })
        
        return results
    
    def create_distillation_chain(self, model_sizes: List[int], 
                                dataloader) -> List[nn.Module]:
        """Create a chain of distilled models of decreasing size"""
        
        print(f"⛓️ Creating distillation chain with {len(model_sizes)} models...")
        
        distilled_models = [self.teacher]  # Start with teacher
        
        current_teacher = self.teacher
        
        for i, size_factor in enumerate(model_sizes):
            print(f"   Step {i+1}: Creating model with size factor {size_factor}")
            
            # Create smaller student model
            student = self._create_student_model(size_factor)
            
            # Update distillation manager with new models
            self.teacher = current_teacher
            self.student = student
            
            # Perform distillation
            self.setup_distillation(self.config)
            self.distill_knowledge(dataloader, num_epochs=3)
            
            # Store distilled model
            distilled_models.append(student)
            
            # This student becomes the next teacher
            current_teacher = student
        
        # Restore original models
        self.teacher = distilled_models[0]
        self.student = distilled_models[-1]
        
        print(f"✅ Distillation chain created with {len(distilled_models)} models")
        return distilled_models
    
    def _setup_distillation_hooks(self):
        """Setup hooks for intermediate feature extraction"""
        
        # This would setup hooks to extract attention maps, hidden states, etc.
        # For now, it's a placeholder
        pass
    
    def _evaluate_student(self, dataloader) -> float:
        """Evaluate student model accuracy"""
        
        self.student.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                outputs = self.student(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    def _evaluate_teacher(self) -> float:
        """Evaluate teacher model accuracy"""
        
        self.teacher.eval()
        # This would use a validation dataset
        # For simulation, return a high accuracy
        return 0.95
    
    def _create_student_model(self, size_factor: float) -> nn.Module:
        """Create a student model with specified size factor"""
        
        # This would create an actual smaller model architecture
        # For simulation, return a simple model
        class SimpleStudent(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(768, 10)  # Simplified
            
            def forward(self, x):
                return self.fc(x)
        
        return SimpleStudent()
    
    def _estimate_speedup(self, compression_ratio: float) -> float:
        """Estimate inference speedup from distillation"""
        
        # Simplified estimation
        base_speedup = compression_ratio * 0.8  # 80% efficiency
        return min(base_speedup, 10.0)  # Cap at 10x
    
    def _estimate_inference_speedup(self) -> float:
        """Estimate actual inference speedup"""
        
        compression_ratio = self.distillation_stats['compression_ratio']
        return self._estimate_speedup(compression_ratio)
    
    def _calculate_memory_reduction(self) -> float:
        """Calculate memory reduction from distillation"""
        
        teacher_memory = self.distillation_stats['teacher_parameters'] * 4 / (1024**2)  # MB
        student_memory = self.distillation_stats['student_parameters'] * 4 / (1024**2)  # MB
        
        if teacher_memory > 0:
            reduction = (teacher_memory - student_memory) / teacher_memory
            return reduction
        return 0.0
    
    def _get_method_details(self, method: DistillationMethod) -> Dict[str, Any]:
        """Get details about distillation method"""
        
        details = {
            DistillationMethod.STANDARD: {
                'description': 'Standard logits-based distillation',
                'complexity': 'low',
                'effectiveness': 'high'
            },
            DistillationMethod.ATTENTION: {
                'description': 'Attention maps transfer',
                'complexity': 'medium', 
                'effectiveness': 'very high'
            },
            DistillationMethod.HIDDEN_STATES: {
                'description': 'Hidden states matching',
                'complexity': 'high',
                'effectiveness': 'high'
            },
            DistillationMethod.RELATIONAL: {
                'description': 'Relational knowledge transfer',
                'complexity': 'medium',
                'effectiveness': 'medium'
            }
        }
        
        return details.get(method, details[DistillationMethod.STANDARD])
