# /tests/verification/test_conceptual_sparse_attention.py

"""
Test script for Conceptual Sparse Attention
"""

import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.modules.conceptual_sparse_attention import ConceptualSparseAttention
from src.conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph

def test_sparse_attention():
    print("Testing Conceptual Sparse Attention...")
    
    # Initialize components
    ckg = ConceptualKnowledgeGraph()
    dim = 512
    num_heads = 8
    
    # Create sparse attention module
    sparse_attn = ConceptualSparseAttention(
        dim=dim,
        num_heads=num_heads,
        ckg=ckg,
        sparsity_ratio=0.2
    )
    
    # Test with different sequence lengths
    test_cases = [
        (1, 50),   # Short sequence
        (2, 256),  # Medium sequence  
        (1, 1024), # Long sequence
        (4, 512),  # Batch with medium sequences
    ]
    
    for batch_size, seq_len in test_cases:
        print(f"\nTesting batch_size={batch_size}, seq_len={seq_len}")
        
        # Create random input
        x = torch.randn(batch_size, seq_len, dim)
        context = {'domain': 'test', 'task': 'evaluation'}
        
        # Forward pass
        with torch.no_grad():
            output, attn_weights = sparse_attn(x, context=context, return_attention_weights=True)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Attention weights shape: {attn_weights.shape}")
        
        # Check sparsity
        attn_density = (attn_weights != 0).float().mean().item()
        sparsity = 1.0 - attn_density
        print(f"Attention sparsity: {sparsity:.3f} ({sparsity*100:.1f}%)")
        
        # Verify output preservation
        input_norm = torch.norm(x, dim=-1).mean()
        output_norm = torch.norm(output, dim=-1).mean()
        preservation_ratio = output_norm / input_norm
        print(f"Signal preservation: {preservation_ratio:.3f}")
        
        assert output.shape == x.shape, "Output shape should match input shape"
        assert 0.1 <= preservation_ratio <= 2.0, "Reasonable signal preservation"
    
    # Performance report
    performance = sparse_attn.get_performance_report()
    print(f"\n=== Performance Report ===")
    for key, value in performance.items():
        print(f"{key}: {value}")
    
    print("\n✅ All tests passed! Conceptual Sparse Attention is working correctly.")

if __name__ == '__main__':
    test_sparse_attention()