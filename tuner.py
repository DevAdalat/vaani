"""
Automatic hyperparameter tuning for BitNet b1.58 training.
"""

import torch
import numpy as np
from typing import Dict, Optional, Any
from dataclasses import dataclass
from collections import deque
import logging


@dataclass
class TunerConfig:
    """Configuration for hyperparameter tuner."""
    # Learning rate bounds
    min_lr: float = 1e-6
    max_lr: float = 1e-2
    lr_increase_factor: float = 1.05
    lr_decrease_factor: float = 0.95
    
    # Threshold (τ) bounds
    min_threshold: float = 0.3
    max_threshold: float = 0.9
    threshold_decrease_step: float = 0.02
    threshold_increase_step: float = 0.01
    target_sparsity_min: float = 0.5
    target_sparsity_max: float = 0.7
    
    # Scale (β) adaptation
    scale_stability_window: int = 100
    scale_adjustment_factor: float = 0.1
    
    # EMA momentum
    initial_momentum: float = 0.9
    final_momentum: float = 0.999
    momentum_transition_steps: int = 10000
    
    # Warmup adjustment
    min_warmup_steps: int = 1000
    max_warmup_steps: int = 10000
    warmup_stability_threshold: float = 0.01
    
    # Gradient monitoring
    grad_clip_percentile: float = 95
    grad_explosion_threshold: float = 10.0
    grad_vanishing_threshold: float = 1e-4
    
    # Decision making
    history_window: int = 100
    patience: int = 10
    aggressive_mode: bool = False
    
    # Logging
    log_interval: int = 10
    verbose: bool = True


class HyperparameterTuner:
    """
    Intelligent hyperparameter tuner for BitNet b1.58 training.
    Automatically adjusts learning rate, quantization thresholds, scales, and other hyperparameters.
    """
    
    def __init__(self, config: TunerConfig, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        """
        Args:
            config: Tuner configuration
            model: BitNet model instance
            optimizer: Optimizer instance
        """
        self.config = config
        self.model = model
        self.optimizer = optimizer
        
        # History tracking
        self.train_loss_history = deque(maxlen=config.history_window)
        self.val_loss_history = deque(maxlen=config.history_window)
        self.grad_norm_history = deque(maxlen=config.history_window)
        self.sparsity_history = deque(maxlen=config.history_window)
        self.lr_history = []
        self.threshold_history = []
        
        # Per-layer tracking
        self.layer_stats = {}
        self.layer_thresholds = {}
        self._initialize_layer_tracking()
        
        # State tracking
        self.step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.warmup_extended = False
        self.current_lr = self._get_current_lr()
        self.current_momentum = config.initial_momentum
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if config.verbose:
            logging.basicConfig(level=logging.INFO)
    
    def _initialize_layer_tracking(self):
        """Initialize per-layer statistics tracking."""
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight_quantizer'):
                self.layer_stats[name] = {
                    'sparsity': deque(maxlen=self.config.history_window),
                    'scale': deque(maxlen=self.config.history_window),
                    'grad_norm': deque(maxlen=self.config.history_window)
                }
                self.layer_thresholds[name] = module.weight_quantizer.threshold.item()
    
    def _get_current_lr(self) -> float:
        """Get current learning rate from optimizer."""
        return self.optimizer.param_groups[0]['lr']
    
    def _set_lr(self, new_lr: float):
        """Set new learning rate in optimizer."""
        new_lr = np.clip(new_lr, self.config.min_lr, self.config.max_lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        self.current_lr = new_lr
        self.lr_history.append((self.step, new_lr))
    
    def _compute_gradient_stats(self) -> Dict[str, float]:
        """Compute gradient statistics."""
        grad_norms = []
        layer_grad_norms = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                
                # Track per-layer gradient norms
                layer_name = '.'.join(name.split('.')[:-1])
                if layer_name in self.layer_stats:
                    if layer_name not in layer_grad_norms:
                        layer_grad_norms[layer_name] = []
                    layer_grad_norms[layer_name].append(grad_norm)
        
        if not grad_norms:
            return {'global_norm': 0.0, 'max_norm': 0.0, 'mean_norm': 0.0}
        
        # Update per-layer stats
        for layer_name, norms in layer_grad_norms.items():
            self.layer_stats[layer_name]['grad_norm'].append(np.mean(norms))
        
        return {
            'global_norm': np.linalg.norm(grad_norms),
            'max_norm': max(grad_norms),
            'mean_norm': np.mean(grad_norms),
            'percentile_95': np.percentile(grad_norms, self.config.grad_clip_percentile)
        }
    
    def _analyze_loss_trend(self) -> str:
        """Analyze recent loss trend."""
        if len(self.train_loss_history) < 10:
            return 'insufficient_data'
        
        recent_losses = list(self.train_loss_history)[-10:]
        
        # Check for explosion
        if recent_losses[-1] > 2 * np.mean(recent_losses[:-1]):
            return 'exploding'
        
        # Check for plateau
        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)
        if loss_std / (loss_mean + 1e-8) < 0.01:
            return 'plateau'
        
        # Check for improvement
        first_half = np.mean(recent_losses[:5])
        second_half = np.mean(recent_losses[5:])
        if second_half < first_half * 0.95:
            return 'improving'
        
        return 'stable'
    
    def _adjust_learning_rate(self, grad_stats: Dict[str, float], loss_trend: str):
        """Adjust learning rate based on gradients and loss trend."""
        current_lr = self.current_lr
        new_lr = current_lr
        reason = ""
        
        # Check for gradient explosion
        if grad_stats['max_norm'] > self.config.grad_explosion_threshold:
            new_lr = current_lr * self.config.lr_decrease_factor
            reason = f"Gradient explosion detected (max_norm={grad_stats['max_norm']:.2f})"
        
        # Check for gradient vanishing
        elif grad_stats['mean_norm'] < self.config.grad_vanishing_threshold:
            new_lr = current_lr * self.config.lr_increase_factor
            reason = f"Gradient vanishing detected (mean_norm={grad_stats['mean_norm']:.2e})"
        
        # Loss-based adjustments
        elif loss_trend == 'exploding':
            new_lr = current_lr * self.config.lr_decrease_factor ** 2
            reason = "Loss explosion detected"
        
        elif loss_trend == 'plateau':
            if self.patience_counter >= self.config.patience:
                new_lr = current_lr * self.config.lr_decrease_factor
                reason = f"Loss plateau for {self.patience_counter} steps"
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        
        elif loss_trend == 'improving':
            self.patience_counter = 0
            if self.config.aggressive_mode:
                new_lr = current_lr * self.config.lr_increase_factor
                reason = "Aggressive mode: increasing LR on improvement"
        
        if new_lr != current_lr:
            self._set_lr(new_lr)
            if self.config.verbose and reason:
                self.logger.info(f"Step {self.step}: LR adjusted from {current_lr:.2e} to {new_lr:.2e}. Reason: {reason}")
    
    def _adjust_thresholds(self, sparsity_stats: Dict[str, float]):
        """Adjust ternary quantization thresholds based on sparsity."""
        global_sparsity = np.mean(list(sparsity_stats.values()))
        
        # Global threshold adjustment
        if global_sparsity < self.config.target_sparsity_min:
            # Too few zeros, increase threshold
            adjustment = self.config.threshold_increase_step
            reason = f"Sparsity too low ({global_sparsity:.2%})"
        elif global_sparsity > self.config.target_sparsity_max:
            # Too many zeros, decrease threshold
            adjustment = -self.config.threshold_decrease_step
            reason = f"Sparsity too high ({global_sparsity:.2%})"
        else:
            adjustment = 0
            reason = ""
        
        if adjustment != 0:
            # Apply per-layer adjustments
            for name, module in self.model.named_modules():
                if hasattr(module, 'weight_quantizer'):
                    current_threshold = module.weight_quantizer.threshold.item()
                    
                    # Per-layer adjustment based on local statistics
                    if name in sparsity_stats:
                        layer_sparsity = sparsity_stats[name]
                        
                        # More aggressive adjustment for outlier layers
                        if layer_sparsity < self.config.target_sparsity_min * 0.8:
                            layer_adjustment = adjustment * 1.5
                        elif layer_sparsity > self.config.target_sparsity_max * 1.2:
                            layer_adjustment = adjustment * 1.5
                        else:
                            layer_adjustment = adjustment
                    else:
                        layer_adjustment = adjustment
                    
                    new_threshold = np.clip(
                        current_threshold + layer_adjustment,
                        self.config.min_threshold,
                        self.config.max_threshold
                    )
                    
                    module.weight_quantizer.update_threshold(new_threshold)
                    self.layer_thresholds[name] = new_threshold
            
            if self.config.verbose and reason:
                self.logger.info(f"Step {self.step}: Threshold adjusted by {adjustment:.3f}. Reason: {reason}")
    
    def _adjust_momentum(self):
        """Adjust EMA momentum for activation quantizers."""
        # Linear schedule from initial to final momentum
        progress = min(self.step / self.config.momentum_transition_steps, 1.0)
        new_momentum = (
            self.config.initial_momentum +
            (self.config.final_momentum - self.config.initial_momentum) * progress
        )
        
        if abs(new_momentum - self.current_momentum) > 0.01:
            for module in self.model.modules():
                if hasattr(module, 'activation_quantizer'):
                    module.activation_quantizer.update_momentum(new_momentum)
            
            self.current_momentum = new_momentum
            if self.config.verbose:
                self.logger.info(f"Step {self.step}: EMA momentum adjusted to {new_momentum:.3f}")
    
    def _check_warmup_completion(self, grad_stats: Dict[str, float]) -> bool:
        """Check if warmup phase can be completed early."""
        if len(self.grad_norm_history) < 50:
            return False
        
        # Check gradient stability
        recent_grads = list(self.grad_norm_history)[-50:]
        grad_std = np.std(recent_grads)
        grad_mean = np.mean(recent_grads)
        
        is_stable = (grad_std / (grad_mean + 1e-8)) < self.config.warmup_stability_threshold
        
        # Check loss stability
        if len(self.train_loss_history) >= 50:
            recent_losses = list(self.train_loss_history)[-50:]
            loss_std = np.std(recent_losses)
            loss_mean = np.mean(recent_losses)
            is_stable = is_stable and (loss_std / (loss_mean + 1e-8)) < self.config.warmup_stability_threshold
        
        return is_stable
    
    def update(
        self,
        train_loss: float,
        val_loss: Optional[float] = None,
        model_stats: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main update function called after each evaluation step.
        
        Args:
            train_loss: Current training loss
            val_loss: Current validation loss (if available)
            model_stats: Additional model statistics
            
        Returns:
            Dictionary of tuning decisions and statistics
        """
        self.step += 1
        
        # Update history
        self.train_loss_history.append(train_loss)
        if val_loss is not None:
            self.val_loss_history.append(val_loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
        
        # Compute gradient statistics
        grad_stats = self._compute_gradient_stats()
        self.grad_norm_history.append(grad_stats['global_norm'])
        
        # Extract sparsity statistics
        sparsity_stats = {}
        if model_stats and 'sparsity' in model_stats:
            for name, module in self.model.named_modules():
                if hasattr(module, 'weight_stats'):
                    sparsity = module.weight_stats[0].item()
                    sparsity_stats[name] = sparsity
                    if name in self.layer_stats:
                        self.layer_stats[name]['sparsity'].append(sparsity)
        
        # Analyze trends
        loss_trend = self._analyze_loss_trend()
        
        # Make adjustments
        decisions = {
            'step': self.step,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'loss_trend': loss_trend,
            'grad_norm': grad_stats['global_norm'],
            'adjustments': []
        }
        
        # Learning rate adjustment
        self._adjust_learning_rate(grad_stats, loss_trend)
        decisions['lr'] = self.current_lr
        
        # Threshold adjustment
        if sparsity_stats:
            self._adjust_thresholds(sparsity_stats)
            decisions['avg_sparsity'] = np.mean(list(sparsity_stats.values()))
        
        # Momentum adjustment
        self._adjust_momentum()
        decisions['momentum'] = self.current_momentum
        
        # Log decisions periodically
        if self.step % self.config.log_interval == 0 and self.config.verbose:
            self.logger.info(f"Step {self.step}: loss={train_loss:.4f}, lr={self.current_lr:.2e}, "
                           f"grad_norm={grad_stats['global_norm']:.2f}, trend={loss_trend}")
        
        return decisions
    
    def get_current_hyperparams(self) -> Dict[str, Any]:
        """Get current hyperparameter values."""
        return {
            'lr': self.current_lr,
            'momentum': self.current_momentum,
            'thresholds': dict(self.layer_thresholds),
            'step': self.step
        }
    
    def get_history(self) -> Dict[str, Any]:
        """Get tuning history for visualization."""
        return {
            'train_loss': list(self.train_loss_history),
            'val_loss': list(self.val_loss_history),
            'grad_norm': list(self.grad_norm_history),
            'lr_history': self.lr_history,
            'threshold_history': self.threshold_history
        }
