"""
BitLinear and transformer layers for BitNet b1.58 implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from quant import TernaryQuantizer, ActivationQuantizer


class BitLinear(nn.Module):
    """
    BitLinear layer with ternary weights (W1.58) and 8-bit activations (A8).
    Replaces standard nn.Linear with quantization-aware implementation.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        threshold: float = 0.7,
        act_bits: int = 8,
        act_momentum: float = 0.99
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            bias: Whether to use bias (typically False for BitNet)
            threshold: Initial ternary quantization threshold
            act_bits: Activation quantization bits
            act_momentum: EMA momentum for activation quantizer
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weight with proper scaling
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, std=math.sqrt(2.0 / in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Quantizers
        self.weight_quantizer = TernaryQuantizer(threshold=threshold)
        self.activation_quantizer = ActivationQuantizer(bits=act_bits, momentum=act_momentum)
        
        # Layer normalization for activation
        self.act_norm = nn.LayerNorm(in_features)
        
        # Statistics tracking
        self.register_buffer('weight_stats', torch.zeros(4))
        self.register_buffer('act_stats', torch.zeros(4))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantization.
        
        Args:
            x: Input tensor of shape (batch, seq_len, in_features)
            
        Returns:
            Output tensor of shape (batch, seq_len, out_features)
        """
        # Normalize and quantize activations
        x = self.act_norm(x)
        x_quant, act_stats = self.activation_quantizer(x)
        
        # Quantize weights
        w_quant, weight_stats = self.weight_quantizer(self.weight)
        
        # Linear transformation
        output = F.linear(x_quant, w_quant, self.bias)
        
        # Store statistics for monitoring
        if self.training:
            self.weight_stats[0] = weight_stats.get('sparsity', 0)
            self.weight_stats[1] = weight_stats.get('scale', 1)
            self.act_stats[0] = act_stats.get('scale', 1)
            self.act_stats[1] = act_stats.get('saturation_rate', 0)
        
        return output
    
    def update_quantization_params(self, threshold: Optional[float] = None, 
                                  act_momentum: Optional[float] = None):
        """Update quantization hyperparameters."""
        if threshold is not None:
            self.weight_quantizer.update_threshold(threshold)
        if act_momentum is not None:
            self.activation_quantizer.update_momentum(act_momentum)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    More efficient than LayerNorm for large models.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with BitLinear layers and RoPE embeddings.
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        threshold: float = 0.7
    ):
        super().__init__()
        assert dim % n_heads == 0
        
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        # BitLinear projections
        self.q_proj = BitLinear(dim, dim, bias=False, threshold=threshold)
        self.k_proj = BitLinear(dim, dim, bias=False, threshold=threshold)
        self.v_proj = BitLinear(dim, dim, bias=False, threshold=threshold)
        self.out_proj = BitLinear(dim, dim, bias=False, threshold=threshold)
        
        self.dropout = nn.Dropout(dropout)
        
        # RoPE embeddings
        self.register_buffer('rope_freqs', self._compute_rope_freqs(max_seq_len))
    
    def _compute_rope_freqs(self, max_seq_len: int) -> torch.Tensor:
        """Compute RoPE frequency tensor."""
        theta = 10000.0
        dim = self.head_dim
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, freqs)
        return torch.polar(torch.ones_like(freqs), freqs)
    
    def _apply_rope(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Apply rotary position embeddings."""
        seq_len = x.shape[1]
        x_complex = torch.view_as_complex(
            x.float().reshape(*x.shape[:-1], -1, 2)
        )
        freqs = self.rope_freqs[offset:offset+seq_len]
        x_rotated = x_complex * freqs.unsqueeze(0).unsqueeze(2)
        return torch.view_as_real(x_rotated).reshape(*x.shape).type_as(x)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Attention mask
            cache: KV cache for inference
            
        Returns:
            Output tensor and updated cache
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute QKV projections
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        offset = 0
        if cache is not None:
            k_cache, v_cache = cache
            offset = k_cache.shape[2]
            q = self._apply_rope(q, offset)
            k = self._apply_rope(k, offset)
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        else:
            q = self._apply_rope(q)
            k = self._apply_rope(k)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        # Update cache
        new_cache = (k, v) if cache is not None else None
        
        return output, new_cache
    
    def update_quantization_params(self, threshold: Optional[float] = None,
                                  act_momentum: Optional[float] = None):
        """Update quantization parameters for all BitLinear layers."""
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            layer.update_quantization_params(threshold, act_momentum)


class FeedForward(nn.Module):
    """
    Feed-forward network with BitLinear layers and SwiGLU activation.
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        threshold: float = 0.7
    ):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        
        # SwiGLU requires 3 projections
        self.gate_proj = BitLinear(dim, hidden_dim, bias=False, threshold=threshold)
        self.up_proj = BitLinear(dim, hidden_dim, bias=False, threshold=threshold)
        self.down_proj = BitLinear(hidden_dim, dim, bias=False, threshold=threshold)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with SwiGLU activation.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.dropout(x)
        x = self.down_proj(x)
        return x
    
    def update_quantization_params(self, threshold: Optional[float] = None,
                                  act_momentum: Optional[float] = None):
        """Update quantization parameters for all BitLinear layers."""
        for layer in [self.gate_proj, self.up_proj, self.down_proj]:
            layer.update_quantization_params(threshold, act_momentum)


class TransformerBlock(nn.Module):
    """
    Transformer block with BitLinear layers, RMSNorm, and residual connections.
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        threshold: float = 0.7
    ):
        super().__init__()
        self.attention = MultiHeadAttention(dim, n_heads, max_seq_len, dropout, threshold)
        self.feed_forward = FeedForward(dim, dropout=dropout, threshold=threshold)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of transformer block.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Attention mask
            cache: KV cache for inference
            
        Returns:
            Output tensor and updated cache
        """
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, new_cache = self.attention(normed, mask, cache)
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual
        normed = self.norm2(x)
        ff_out = self.feed_forward(normed)
        x = x + self.dropout(ff_out)
        
        return x, new_cache
    
    def update_quantization_params(self, threshold: Optional[float] = None,
                                  act_momentum: Optional[float] = None):
        """Update quantization parameters for all sublayers."""
        self.attention.update_quantization_params(threshold, act_momentum)
        self.feed_forward.update_quantization_params(threshold, act_momentum)
