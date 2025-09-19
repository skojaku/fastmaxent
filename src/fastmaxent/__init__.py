"""
FastMaxEnt: Fast Unbiased Sampling of Networks with Given Expected Degrees and Strengths

This package provides efficient algorithms for sampling unweighted and weighted networks
from the Undirected Binary Configuration Model (UBCM) and Undirected Enhanced Configuration
Model (UECM) respectively.

The implementation uses rejection sampling with geometric jumps to achieve fast sampling
times while maintaining statistical accuracy.

Additionally, this package provides fast parameter inference capabilities using Adam
optimization with Numba acceleration.
"""

from .fastmaxent import sampling
from .inference import (
    calc_grad,
    calc_grad_fast,
    estimate_parameters, 
    validate_parameters,
    initialize_parameters
)
from .optimizer import AdamOptimizer, clip_grad_norm, schedule_lr

__version__ = "0.1.0"
__author__ = "Xuanchi Li, Xin Wang, Sadamori Kojaku"

__all__ = [
    "sampling",
    "calc_grad",
    "calc_grad_fast", 
    "estimate_parameters",
    "validate_parameters", 
    "initialize_parameters",
    "AdamOptimizer",
    "clip_grad_norm",
    "schedule_lr"
]