"""
FastMaxEnt: Fast Unbiased Sampling of Networks with Given Expected Degrees and Strengths

This package provides efficient algorithms for sampling unweighted and weighted networks
from the Undirected Binary Configuration Model (UBCM) and Undirected Enhanced Configuration
Model (UECM) respectively.

The implementation uses rejection sampling with geometric jumps to achieve fast sampling
times while maintaining statistical accuracy.
"""

from .fastmaxent import sampling

__version__ = "0.1.0"
__author__ = "Xuanchi Li, Xin Wang, Sadamori Kojaku"

__all__ = ["sampling"]