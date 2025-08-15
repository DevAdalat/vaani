"""
Quantization utilities for BitNet b1.58 (1.58-bit) implementation.
Implements ternary weight quantization and 8-bit activation quantization.
"""

import torch
import torch.nn as nn
from typing import Tuple


class TernaryQuantizer(nn.Module):
    """
    Ternary weight quantizer for BitNet b1.58.
    Quantizes weights to {-1, 0, +1} based on magnitude threshold.
    """
    
    def __init__(self, threshold: float = 0.7):
        """
        Args:
            threshold: Initial threshold τ for ternary quantization (fraction of max |w|)
        """
        super().__init__()
        self.register_buffer('threshold', torch.tensor(threshold))
        self.register_buffer('scale', torch.tensor(1.0))
    
    def forward(self, weight: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Quantize weights to ternary values.
        
        Args:
            weight: Input weight tensor
            
        Returns:
            Quantized weights and statistics dict
        """
        if not self.training and hasattr(self, '_quantized_weight'):
            # Use cached quantized weights during inference
            return self._quantized_weight, {}
        
        # Compute per-channel statistics for better quantization
        if weight.dim() >= 2:
            # Reshape to (out_features, -1) for per-output-channel stats
            w_reshape = weight.view(weight.size(0), -1)
            w_abs_mean = w_reshape.abs().mean(dim=1, keepdim=True)
            w_abs_mean = w_abs_mean.view(-1, *([1] * (weight.dim() - 1)))
        else:
            w_abs_mean = weight.abs().mean()
        
        # Dynamic threshold based on weight statistics
        threshold_value = self.threshold * w_abs_mean
        
        # Ternary quantization
        weight_ternary = torch.sign(weight)
        weight_ternary = weight_ternary * (weight.abs() > threshold_value).float()
        
        # Compute optimal scale using L2 norm matching
        if weight.numel() > 0:
            numerator = (weight * weight_ternary).sum()
            denominator = (weight_ternary * weight_ternary).sum().clamp(min=1e-6)
            self.scale = (numerator / denominator).detach()
        
        # Apply scale
        weight_quantized = weight_ternary * self.scale
        
        # Statistics for monitoring
        stats = {
            'sparsity': (weight_ternary == 0).float().mean().item(),
            'pos_ratio': (weight_ternary == 1).float().mean().item(),
            'neg_ratio': (weight_ternary == -1).float().mean().item(),
            'scale': self.scale.item(),
            'threshold': self.threshold.item()
        }
        
        # Cache for inference
        if not self.training:
            self._quantized_weight = weight_quantized
        
        return weight_quantized, stats
    
    def update_threshold(self, new_threshold: float):
        """Update the ternary threshold τ."""
        self.threshold.fill_(new_threshold)
        if hasattr(self, '_quantized_weight'):
            delattr(self, '_quantized_weight')


class ActivationQuantizer(nn.Module):
    """
    8-bit activation quantizer with learnable scale parameters.
    Uses absmax quantization with EMA tracking for stability.
    """
    
    def __init__(self, bits: int = 8, momentum: float = 0.99):
        """
        Args:
            bits: Number of quantization bits
            momentum: EMA momentum for tracking activation scales
        """
        super().__init__()
        self.bits = bits
        self.momentum = momentum
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('running_max', torch.tensor(0.0))
        self.register_buffer('initialized', torch.tensor(False))
        
        # Quantization range
        self.qmin = -(2 ** (bits - 1))
        self.qmax = 2 ** (bits - 1) - 1
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Quantize activations to 8-bit integers.
        
        Args:
            x: Input activation tensor
            
        Returns:
            Quantized activations and statistics dict
        """
        if self.training:
            # Track activation range with EMA
            abs_max = x.abs().max().detach()
            if not self.initialized:
                self.running_max.copy_(abs_max)
                self.initialized.fill_(True)
            else:
                self.running_max.mul_(self.momentum).add_(abs_max, alpha=1 - self.momentum)
            
            scale = self.running_max / self.qmax
        else:
            scale = self.scale if self.initialized else x.abs().max() / self.qmax
        
        scale = scale.clamp(min=1e-6)
        self.scale.copy_(scale)
        
        # Quantize and dequantize (fake quantization for training)
        if self.training:
            x_quant = torch.clamp(torch.round(x / scale), self.qmin, self.qmax)
            x_dequant = x_quant * scale
            
            # Straight-through estimator
            x_out = x + (x_dequant - x).detach()
        else:
            # True quantization for inference
            x_out = torch.clamp(torch.round(x / scale), self.qmin, self.qmax) * scale
        
        stats = {
            'scale': scale.item(),
            'running_max': self.running_max.item(),
            'actual_max': x.abs().max().item(),
            'saturation_rate': ((x.abs() / scale) > self.qmax).float().mean().item()
        }
        
        return x_out, stats
    
    def update_momentum(self, new_momentum: float):
        """Update EMA momentum for activation scale tracking."""
        self.momentum = new_momentum
