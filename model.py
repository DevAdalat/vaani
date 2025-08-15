"""
BitNet b1.58 Language Model implementation.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any
from layers import TransformerBlock, RMSNorm, BitLinear


class Vaani(nn.Module):
    """
    BitNet b1.58 Language Model with ternary weights and 8-bit activations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Model configuration dictionary containing:
                - vocab_size: Vocabulary size
                - dim: Model dimension
                - n_layers: Number of transformer layers
                - n_heads: Number of attention heads
                - max_seq_len: Maximum sequence length
                - dropout: Dropout rate
                - threshold: Initial ternary quantization threshold
        """
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_emb = nn.Embedding(config['vocab_size'], config['dim'])
        nn.init.normal_(self.token_emb.weight, std=0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=config['dim'],
                n_heads=config['n_heads'],
                max_seq_len=config['max_seq_len'],
                dropout=config.get('dropout', 0.1),
                threshold=config.get('threshold', 0.7)
            )
            for _ in range(config['n_layers'])
        ])
        
        # Final norm and output projection
        self.norm = RMSNorm(config['dim'])
        self.output = BitLinear(
            config['dim'],
            config['vocab_size'],
            bias=False,
            threshold=config.get('threshold', 0.7)
        )
        
        # Initialize output weights tied to embeddings if specified
        if config.get('tie_embeddings', True):
            self.output.weight = self.token_emb.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the language model.
        
        Args:
            input_ids: Input token IDs (batch, seq_len)
            mask: Attention mask
            cache: KV cache for each layer (for inference)
            
        Returns:
            Dictionary containing:
                - logits: Output logits (batch, seq_len, vocab_size)
                - cache: Updated KV cache
                - stats: Model statistics
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.token_emb(input_ids)
        
        # Create causal mask if not provided
        if mask is None and self.training:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            mask = mask.to(x.device)
            mask = mask.unsqueeze(0).unsqueeze(1)
        
        # Process through transformer blocks
        new_cache = []
        stats = {'sparsity': [], 'scales': []}
        
        for i, block in enumerate(self.blocks):
            layer_cache = cache[i] if cache is not None else None
            x, updated_cache = block(x, mask, layer_cache)
            
            if cache is not None:
                new_cache.append(updated_cache)
            
            # Collect statistics
            if self.training:
                with torch.no_grad():
                    # Collect sparsity from attention layers
                    for name, module in block.named_modules():
                        if hasattr(module, 'weight_stats'):
                            stats['sparsity'].append(module.weight_stats[0].item())
                            stats['scales'].append(module.weight_stats[1].item())
        
        # Final norm and output projection
        x = self.norm(x)
        logits = self.output(x)
        
        return {
            'logits': logits,
            'cache': new_cache if cache is not None else None,
            'stats': stats
        }
    
    def update_quantization_params(
        self,
        threshold: Optional[float] = None,
        act_momentum: Optional[float] = None,
        layer_thresholds: Optional[List[float]] = None
    ):
        """
        Update quantization parameters globally or per-layer.
        
        Args:
            threshold: Global threshold for all layers
            act_momentum: Global activation momentum
            layer_thresholds: Per-layer thresholds (overrides global)
        """
        if layer_thresholds is not None:
            for i, (block, layer_threshold) in enumerate(zip(self.blocks, layer_thresholds)):
                block.update_quantization_params(layer_threshold, act_momentum)
        elif threshold is not None or act_momentum is not None:
            for block in self.blocks:
                block.update_quantization_params(threshold, act_momentum)
            
            # Update output layer
            self.output.update_quantization_params(threshold, act_momentum)
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_ternary_params(self) -> int:
        """Get number of ternary (1.58-bit) parameters."""
        count = 0
        for module in self.modules():
            if isinstance(module, BitLinear):
                count += module.weight.numel()
        return count
