"""
Training script for BitNet b1.58 on Tiny Shakespeare dataset with character-level tokenization.
Includes automatic hyperparameter tuning and comprehensive logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
import os
import json
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import time

from model import Vaani
from tuner import HyperparameterTuner, TunerConfig


class CharacterTokenizer:
    """
    Character-level tokenizer for text data.
    """
    
    def __init__(self, text: str):
        """
        Initialize tokenizer with text corpus.
        
        Args:
            text: Input text to build vocabulary from
        """
        # Build vocabulary from unique characters
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        
        # Create mappings
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"First 10 chars: {self.chars[:10]}")
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        return [self.char_to_idx.get(ch, 0) for ch in text]
    
    def decode(self, tokens: List[int]) -> str:
        """Convert token IDs back to text."""
        return ''.join([self.idx_to_char.get(tok, '') for tok in tokens])
    
    def save(self, path: str):
        """Save tokenizer to file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'chars': self.chars,
                'vocab_size': self.vocab_size
            }, f, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str):
        """Load tokenizer from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls.__new__(cls)
        tokenizer.chars = data['chars']
        tokenizer.vocab_size = data['vocab_size']
        tokenizer.char_to_idx = {ch: i for i, ch in enumerate(tokenizer.chars)}
        tokenizer.idx_to_char = {i: ch for i, ch in enumerate(tokenizer.chars)}
        return tokenizer


class ShakespeareDataset(Dataset):
    """
    Dataset for Tiny Shakespeare with character-level tokenization.
    """
    
    def __init__(
        self,
        data: str,
        tokenizer: CharacterTokenizer,
        seq_len: int,
        stride: Optional[int] = None
    ):
        """
        Args:
            data: Raw text data
            tokenizer: Character tokenizer
            seq_len: Sequence length for each sample
            stride: Stride for creating overlapping sequences (defaults to seq_len)
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride or seq_len
        
        # Tokenize entire text
        self.tokens = torch.tensor(tokenizer.encode(data), dtype=torch.long)
        
        # Calculate number of sequences
        self.num_sequences = max(1, (len(self.tokens) - seq_len - 1) // self.stride + 1)
        
        print(f"Dataset: {len(self.tokens):,} tokens, {self.num_sequences:,} sequences")
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence and its target.
        
        Returns:
            (input_sequence, target_sequence) where target is input shifted by 1
        """
        start_idx = idx * self.stride
        end_idx = min(start_idx + self.seq_len + 1, len(self.tokens))
        
        sequence = self.tokens[start_idx:end_idx]
        
        # Pad if necessary
        if len(sequence) < self.seq_len + 1:
            padding = torch.zeros(self.seq_len + 1 - len(sequence), dtype=torch.long)
            sequence = torch.cat([sequence, padding])
        
        return sequence[:-1], sequence[1:]


def download_tiny_shakespeare(cache_dir: str = "./data") -> str:
    """
    Download Tiny Shakespeare dataset.
    
    Args:
        cache_dir: Directory to cache the dataset
        
    Returns:
        Path to downloaded file
    """
    os.makedirs(cache_dir, exist_ok=True)
    file_path = os.path.join(cache_dir, "tiny_shakespeare.txt")
    
    if not os.path.exists(file_path):
        print("Downloading Tiny Shakespeare dataset...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        
        response = requests.get(url)
        response.raise_for_status()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"Dataset downloaded to {file_path}")
    else:
        print(f"Using cached dataset from {file_path}")
    
    return file_path


class Trainer:
    """
    Trainer for BitNet b1.58 on Tiny Shakespeare with automatic hyperparameter tuning.
    """
    
    def __init__(
        self,
        model: Vaani,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        tokenizer: CharacterTokenizer,
        config: Dict[str, Any]
    ):
        """
        Args:
            model: BitNet language model
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            tokenizer: Character tokenizer for decoding
            config: Training configuration
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01),
            betas=(0.9, 0.95)
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.get('warmup_steps', 1000),
            T_mult=2,
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        # Setup hyperparameter tuner
        tuner_config = TunerConfig(
            min_lr=config.get('min_lr', 1e-6),
            max_lr=config.get('max_lr', 1e-2),
            lr_decrease_factor=0.9,
            lr_increase_factor=1.1,
            target_sparsity_min=config.get('target_sparsity_min', 0.5),
            target_sparsity_max=config.get('target_sparsity_max', 0.7),
            aggressive_mode=config.get('aggressive_tuning', False),
            verbose=config.get('verbose', True),
            log_interval=config.get('log_interval', 10)
        )
        self.tuner = HyperparameterTuner(tuner_config, model, self.optimizer)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = []
        self.val_history = []
        
        # Create output directory
        self.output_dir = Path(config.get('output_dir', './outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: (input_ids, targets) tuple
            
        Returns:
            Dictionary of training statistics
        """
        input_ids, targets = batch
        input_ids = input_ids.to(self.device)
        targets = targets.to(self.device)
        
        # Forward pass
        self.model.train()
        outputs = self.model(input_ids)
        logits = outputs['logits']
        
        # Compute loss
        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.get('grad_clip', 1.0)
        )
        
        # Optimizer step
        self.optimizer.step()
        
        # Scheduler step (if not using tuner for LR)
        if not self.config.get('use_tuner_lr', True):
            self.scheduler.step()
        
        # Collect statistics
        stats = {
            'loss': loss.item(),
            'lr': self.optimizer.param_groups[0]['lr'],
            'grad_norm': grad_norm.item(),
            'perplexity': torch.exp(loss).item()
        }
        
        # Add model statistics
        if 'stats' in outputs and outputs['stats']:
            if outputs['stats'].get('sparsity'):
                stats['sparsity'] = np.mean(outputs['stats']['sparsity'])
            if outputs['stats'].get('scales'):
                stats['scale_mean'] = np.mean(outputs['stats']['scales'])
        
        return stats
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation on the validation set.
        
        Returns:
            Dictionary of validation statistics
        """
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        all_sparsities = []
        
        progress_bar = tqdm(self.val_dataloader, desc="Validation", leave=False)
        for batch in progress_bar:
            input_ids, targets = batch
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(input_ids)
            logits = outputs['logits']
            
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            
            # Count non-padding tokens
            non_pad_tokens = (targets != 0).sum().item()
            total_loss += loss.item() * non_pad_tokens
            total_tokens += non_pad_tokens
            
            # Collect sparsity statistics
            if 'stats' in outputs and outputs['stats'].get('sparsity'):
                all_sparsities.extend(outputs['stats']['sparsity'])
        
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = np.exp(avg_loss)
        
        stats = {
            'val_loss': avg_loss,
            'val_perplexity': perplexity
        }
        
        if all_sparsities:
            stats['val_sparsity'] = np.mean(all_sparsities)
        
        return stats
    
    @torch.no_grad()
    def generate_sample(self, prompt: str = "To be or not to be", max_length: int = 200) -> str:
        """
        Generate sample text from the model.
        
        Args:
            prompt: Initial prompt text
            max_length: Maximum generation length
            
        Returns:
            Generated text
        """
        self.model.eval()
        
        # Encode prompt
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        generated = tokens.copy()
        
        # Generate tokens
        for _ in range(max_length):
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs['logits']
                
                # Get next token probabilities
                next_token_logits = logits[0, -1, :]
                
                # Apply temperature
                temperature = self.config.get('generation_temperature', 0.8)
                next_token_logits = next_token_logits / temperature
                
                # Sample from distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                generated.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                # Truncate if too long
                if input_ids.size(1) > self.config.get('max_seq_len', 512):
                    input_ids = input_ids[:, -self.config.get('max_seq_len', 512):]
        
        return self.tokenizer.decode(generated)
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'tuner_state': self.tuner.get_current_hyperparams(),
            'tokenizer_vocab_size': self.tokenizer.vocab_size
        }
        
        path = self.output_dir / filename
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")
        
        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        self.logger.info(f"Loaded checkpoint from {path} (epoch {self.epoch}, step {self.global_step})")
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        if not self.training_history:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Loss curve
        steps = [h['step'] for h in self.training_history]
        train_losses = [h['loss'] for h in self.training_history]
        axes[0, 0].plot(steps, train_losses, label='Train Loss')
        if self.val_history:
            val_steps = [h['step'] for h in self.val_history]
            val_losses = [h['val_loss'] for h in self.val_history]
            axes[0, 0].plot(val_steps, val_losses, label='Val Loss')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Perplexity curve
        train_perplexities = [h.get('perplexity', 0) for h in self.training_history]
        axes[0, 1].plot(steps, train_perplexities, label='Train Perplexity')
        if self.val_history:
            val_perplexities = [h.get('val_perplexity', 0) for h in self.val_history]
            axes[0, 1].plot(val_steps, val_perplexities, label='Val Perplexity')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Perplexity')
        axes[0, 1].set_title('Perplexity')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate curve
        lrs = [h['lr'] for h in self.training_history]
        axes[0, 2].plot(steps, lrs)
        axes[0, 2].set_xlabel('Steps')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].set_title('Learning Rate Schedule')
        axes[0, 2].grid(True)
        
        # Gradient norm
        grad_norms = [h.get('grad_norm', 0) for h in self.training_history]
        axes[1, 0].plot(steps, grad_norms)
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Gradient Norm')
        axes[1, 0].set_title('Gradient Norm')
        axes[1, 0].grid(True)
        
        # Sparsity
        sparsities = [h.get('sparsity', 0) for h in self.training_history if 'sparsity' in h]
        if sparsities:
            sparsity_steps = [h['step'] for h in self.training_history if 'sparsity' in h]
            axes[1, 1].plot(sparsity_steps, sparsities)
            axes[1, 1].set_xlabel('Steps')
            axes[1, 1].set_ylabel('Weight Sparsity')
            axes[1, 1].set_title('Ternary Weight Sparsity')
            axes[1, 1].axhline(y=self.config.get('target_sparsity_min', 0.5), color='r', linestyle='--', label='Target Min')
            axes[1, 1].axhline(y=self.config.get('target_sparsity_max', 0.7), color='r', linestyle='--', label='Target Max')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        # Scale evolution
        scales = [h.get('scale_mean', 1.0) for h in self.training_history if 'scale_mean' in h]
        if scales:
            scale_steps = [h['step'] for h in self.training_history if 'scale_mean' in h]
            axes[1, 2].plot(scale_steps, scales)
            axes[1, 2].set_xlabel('Steps')
            axes[1, 2].set_ylabel('Mean Scale')
            axes[1, 2].set_title('Quantization Scale Evolution')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=150)
        plt.close()
    
    def train(self):
        """Main training loop with automatic hyperparameter tuning."""
        self.logger.info(f"Starting training on {self.device}")
        self.logger.info(f"Model parameters: {self.model.get_num_params():,}")
        self.logger.info(f"Ternary parameters: {self.model.get_num_ternary_params():,}")
        self.logger.info(f"Output directory: {self.output_dir}")
        
        # Training loop
        for epoch in range(self.config['num_epochs']):
            self.epoch = epoch
            epoch_loss = 0.0
            epoch_tokens = 0
            
            # Training epoch
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
            for batch_idx, batch in enumerate(progress_bar):
                self.global_step += 1
                
                # Training step
                train_stats = self.train_step(batch)
                
                # Update epoch statistics
                batch_size = batch[0].size(0) * batch[0].size(1)
                epoch_loss += train_stats['loss'] * batch_size
                epoch_tokens += batch_size
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{train_stats['loss']:.4f}",
                    'ppl': f"{train_stats['perplexity']:.2f}",
                    'lr': f"{train_stats['lr']:.2e}",
                    'sparsity': f"{train_stats.get('sparsity', 0):.1%}"
                })
                
                # Logging
                if self.global_step % self.config['log_interval'] == 0:
                    train_stats['step'] = self.global_step
                    train_stats['epoch'] = epoch
                    self.training_history.append(train_stats)
                
                # Validation and hyperparameter tuning
                if self.global_step % self.config['eval_interval'] == 0:
                    val_stats = self.validate()
                    val_stats['step'] = self.global_step
                    val_stats['epoch'] = epoch
                    self.val_history.append(val_stats)
                    
                    # Update hyperparameters with tuner
                    if self.config.get('use_tuner', True):
                        tuning_decisions = self.tuner.update(
                            train_loss=train_stats['loss'],
                            val_loss=val_stats.get('val_loss'),
                            model_stats=train_stats
                        )
                    
                    # Generate sample text
                    if self.global_step % self.config.get('generation_interval', 1000) == 0:
                        sample = self.generate_sample()
                        self.logger.info(f"\n--- Generated Sample ---\n{sample}\n---")
                    
                    # Log progress
                    self.logger.info(
                        f"Step {self.global_step} | Epoch {epoch} | "
                        f"Train Loss: {train_stats['loss']:.4f} | "
                        f"Val Loss: {val_stats.get('val_loss', 0):.4f} | "
                        f"Val PPL: {val_stats.get('val_perplexity', 0):.2f} | "
                        f"LR: {train_stats['lr']:.2e} | "
                        f"Sparsity: {train_stats.get('sparsity', 0):.1%}"
                    )
                    
                    # Save checkpoint if best
                    if val_stats.get('val_loss', float('inf')) < self.best_val_loss:
                        self.best_val_loss = val_stats['val_loss']
                        self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt', is_best=True)
                
                # Save periodic checkpoint
                if self.global_step % self.config.get('save_interval', 5000) == 0:
                    self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')
                    self.plot_training_curves()
                
                # Early stopping check
                if self.global_step >= self.config.get('max_steps', float('inf')):
                    break
            
            # End of epoch
            avg_epoch_loss = epoch_loss / max(epoch_tokens, 1)
            self.logger.info(f"Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Save epoch checkpoint
            self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
            
            if self.global_step >= self.config.get('max_steps', float('inf')):
                break
        
        # Final saves
        self.save_checkpoint('final_model.pt')
        self.plot_training_curves()
        self.save_training_history()
        
        # Generate final samples
        self.logger.info("\n=== Final Text Generation Samples ===")
        prompts = [
            "To be or not to be",
            "All the world's a stage",
            "Romeo, Romeo, wherefore art thou",
            "Now is the winter of our discontent",
            "Friends, Romans, countrymen"
        ]
        
        for prompt in prompts:
            sample = self.generate_sample(prompt, max_length=200)
            self.logger.info(f"\nPrompt: {prompt}")
            self.logger.info(f"Generated: {sample}\n")
    
    def save_training_history(self):
        """Save complete training history to JSON."""
        history = {
            'config': self.config,
            'training_history': self.training_history,
            'validation_history': self.val_history,
            'tuner_history': self.tuner.get_history(),
            'final_hyperparams': self.tuner.get_current_hyperparams(),
            'best_val_loss': self.best_val_loss,
            'total_steps': self.global_step,
            'total_epochs': self.epoch
        }
        
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        self.logger.info(f"Saved training history to {self.output_dir / 'training_history.json'}")


def main():
    """Main training script for Tiny Shakespeare."""
    
    # Configuration
    model_config = {
        'vocab_size': 65,  # Will be updated based on actual vocabulary
        'dim': 384,        # Model dimension
        'n_layers': 6,     # Number of transformer layers
        'n_heads': 6,      # Number of attention heads
        'max_seq_len': 256,  # Maximum sequence length
        'dropout': 0.1,
        'threshold': 0.7,  # Initial ternary quantization threshold
        'tie_embeddings': True
    }
    
    train_config = {
        # Data
        'batch_size': 64,
        'seq_len': 256,
        'train_split': 0.9,
        
        # Training
        'num_epochs': 10,
        'max_steps': 50000,
        'learning_rate': 6e-4,
        'min_lr': 1e-6,
        'max_lr': 1e-2,
        'weight_decay': 0.1,
        'grad_clip': 1.0,
        'warmup_steps': 1000,
        
        # Evaluation
        'eval_interval': 500,
        'log_interval': 50,
        'save_interval': 5000,
        'generation_interval': 1000,
        'generation_temperature': 0.8,
        
        # Hyperparameter tuning
        'use_tuner': True,
        'use_tuner_lr': True,
        'aggressive_tuning': False,
        'target_sparsity_min': 0.5,
        'target_sparsity_max': 0.7,
        
        # Output
        'output_dir': './outputs/shakespeare_bitnet',
        'verbose': True
    }
    
    # Download and load dataset
    data_path = download_tiny_shakespeare()
    
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Loaded {len(text):,} characters of text")
    print(f"First 100 characters: {text[:100]}")
    
    # Create tokenizer
    tokenizer = CharacterTokenizer(text)
    
    # Update model config with actual vocab size
    model_config['vocab_size'] = tokenizer.vocab_size
    
    # Split data
    split_idx = int(len(text) * train_config['train_split'])
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    print(f"Train: {len(train_text):,} chars, Val: {len(val_text):,} chars")
    
    # Create datasets
    train_dataset = ShakespeareDataset(
        train_text,
        tokenizer,
        seq_len=train_config['seq_len']
    )
    
    val_dataset = ShakespeareDataset(
        val_text,
        tokenizer,
        seq_len=train_config['seq_len']
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Create model
    model = Vaani(model_config)
    
    print(f"\nModel Configuration:")
    print(f"  Total parameters: {model.get_num_params():,}")
    print(f"  Ternary parameters: {model.get_num_ternary_params():,}")
    print(f"  Model dimension: {model_config['dim']}")
    print(f"  Layers: {model_config['n_layers']}")
    print(f"  Attention heads: {model_config['n_heads']}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer,
        config={**model_config, **train_config}
    )
    
    # Start training
    start_time = time.time()
    trainer.train()
    
    # Training complete
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time/3600:.2f} hours")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Best validation perplexity: {np.exp(trainer.best_val_loss):.2f}")


if __name__ == "__main__":
    main()