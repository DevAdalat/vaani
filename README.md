# Vaani: A BitNet b1.58 Language Model

## Introduction

Vaani is an efficient and cutting-edge language model implementation based on the **BitNet b1.58** architecture. This project showcases how to build and train a highly optimized neural network that leverages extremely low-bit quantization for both weights and activations. By utilizing 1.58-bit (ternary) weights and 8-bit activations, Vaani significantly reduces memory footprint and computational requirements compared to traditional full-precision models, making it ideal for deployment on resource-constrained devices or for scaling up large models efficiently.

This implementation is designed to be a clear and comprehensive example, allowing researchers and developers to understand, experiment with, and build upon the principles of BitNet b1.58.

## Features

- **1.58-bit Ternary Weights**: Implemented via the custom `BitLinear` layer, weights are quantized to {-1, 0, +1}, drastically reducing model size.
- **8-bit Quantized Activations**: Activations are quantized to 8-bit integers, further enhancing memory and computational efficiency.
- **Modern Transformer Architecture**: Built upon the robust Transformer framework, incorporating:
  - **`MultiHeadAttention` with Rotary Position Embeddings (RoPE)**: An advanced positional encoding technique for improved sequence understanding.
  - **SwiGLU `FeedForward` Networks**: A high-performing activation function known for its effectiveness in modern language models.
  - **`RMSNorm`**: An efficient alternative to Layer Normalization.
- **Quantization-Aware Training (QAT) Components**: Includes `TernaryQuantizer` and `ActivationQuantizer` to facilitate training with low-bit precision.
- **Automated Hyperparameter Tuning**: The `HyperparameterTuner` (from `tuner.py`) intelligently adjusts learning rates, quantization thresholds, and other parameters during training to optimize performance.
- **Comprehensive Training Script**: The `train.py` script provides a full training pipeline, including data loading, model definition, training loop, evaluation, and checkpointing.

## Getting Started

To get started with Vaani, follow these steps to set up your environment and install the necessary dependencies.

### Prerequisites

- Python 3.8+
- PyTorch (compatible with your CUDA version if you plan to use GPU)

### Installation

1.  Clone this repository:

    ```bash
    git clone https://github.com/DevAdalat/vaani.git
    cd vaani
    ```

2.  Install the required Python packages using the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Training the Model

The `train.py` script is the entry point for training the Vaani model. It includes data preparation (using the Tiny Shakespeare dataset as an example), model instantiation, and the training loop with integrated hyperparameter tuning.

### Data

The `train.py` script will automatically download the Tiny Shakespeare dataset if it's not already present. This dataset is used for character-level language modeling.

### Configuration

The training process is configured via `model_config` and `train_config` dictionaries within `train.py`. You can modify these to experiment with different model sizes, training parameters, and tuning strategies.

Key configurable parameters include:

- `vocab_size`, `dim`, `n_layers`, `n_heads`, `max_seq_len`, `dropout`, `threshold` (for the model architecture)
- `batch_size`, `num_epochs`, `learning_rate`, `eval_interval`, `save_interval` (for training)
- `use_tuner`, `aggressive_tuning`, `target_sparsity_min`, `target_sparsity_max` (for hyperparameter tuning)

### Running Training

To start training, simply run the `train.py` script:

```bash
python train.py
```

The script will output training progress, validation metrics, and periodically save model checkpoints and training history.

## Performance Notes

We have successfully trained a small-sized version of this model and observed promising accuracy. This indicates the effectiveness of the BitNet b1.58 architecture and the quality of this implementation. With sufficient resources, training larger models based on this codebase is expected to yield excellent results.

_Note from the owner_: I am not an expert in AI/ML, but this project represents my best effort to make a highly efficient model publicly available. I have observed good accuracy with a small-sized model trained using this code. I am also actively working on a dataset cleaner model, which I believe will further enhance the accuracy of models trained with this codebase.

## Credit and Acknowledgements

This project is built upon the foundational research of BitNet b1.58. We extend our gratitude to the original authors and researchers whose work made this efficient architecture possible.

If you use this code or build upon it, please consider citing the original BitNet b1.58 paper and acknowledging this repository.

**Original Owner/Contributors**:
@DevAdalat - [https://github.com/DevAdalat](https://github.com/DevAdalat)

## License

This project is open-sourced under the MIT License. See the `LICENSE` file for more details.
