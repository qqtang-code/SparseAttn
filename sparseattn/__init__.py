"""
Sparse Attention Library
=========================

A high-performance sparse attention library for large language models.

This library provides efficient implementations of various sparse attention mechanisms
designed to reduce memory usage and computation time in large language models.

Core Modules:
-------------
- Xattention: Adaptive sparse attention based on thresholding
- FlexPrefill: Block-level sparse attention with adaptive selection
- Minference: Lightweight inference with vertical and diagonal sparsity
- FullPrefill: Complete prefill implementation based on FlashInfer

Threshold-based Modules:
------------------------
- llama_fuse_16: LLaMA threshold fusion for 16-bit precision
- llama_fuse_8: LLaMA threshold fusion for 8-bit precision
- llama_fuse_4: LLaMA threshold fusion for 4-bit precision

Training Modules:
-----------------
- DistributedAttention: Distributed attention implementation for multi-GPU training

Examples:
---------
>>> from sparseattn import Flexprefill_prefill
>>> from sparseattn import llama_fuse_16
>>> from sparseattn import DistributedAttention

For more information, visit: https://github.com/qqtang-code/SparseAttn
"""

# Core sparse attention implementations
from .src import (
    Xattention_prefill_dim3,
    Xattention_prefill_dim4,
    Flexprefill_prefill,
    Minference_prefill,
    Full_prefill,
    load_LLM,
)

# Threshold-based sparse attention modules
from .threshold import (
    llama_fuse_16,
    llama_fuse_8,
    llama_fuse_4,
)

# Training modules
from .training import DistributedAttention

# For backward compatibility and ease of use
Xattention = Xattention_prefill_dim4
FlexPrefill = Flexprefill_prefill
Minference = Minference_prefill
FullPrefill = Full_prefill
LoadLLM = load_LLM

__all__ = [
    # Core modules
    "Xattention_prefill",
    "Flexprefill_prefill",
    "Minference_prefill",
    "Full_prefill",
    "load_LLM"
    # Threshold modules
    "llama_fuse_16",
    "llama_fuse_8",
    "llama_fuse_4",
    # Training modules
    "DistributedAttention",
    # Aliases for backward compatibility
    "Xattention",
    "FlexPrefill",
    "Minference",
    "FullPrefill",
    "LoadLLM",
]

__version__ = "0.1.0"
__author__ = "QQTang Code"
