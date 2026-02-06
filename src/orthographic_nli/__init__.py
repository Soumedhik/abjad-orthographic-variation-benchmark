"""
Orthographic Variation Benchmarking for Natural Language Inference.

This package evaluates LLM robustness to realistic orthographic variation
in Arabic, Urdu, Swahili, and English using modified XNLI benchmarks.

Key modules:
    - data: Load and prepare XNLI datasets
    - variants: Generate orthographic perturbations (romanization, code-switching)
    - evaluate: Run model inference and compute metrics
    - groq_client: Interface to Groq API for model inference
    - metrics: Calculate performance deltas
    - config: Configuration management
"""

__version__ = "1.0.0"
__author__ = "Shibam Mandal, Soumedhik Bharati, Swarup Kr Ghosh, Sayani Mondal"
