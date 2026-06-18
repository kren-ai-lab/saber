"""
Core module of mlcore.

This module exposes the main building blocks of the framework:
- Algorithm specification
- Registry system
- Training orchestration
"""

from .specs import AlgorithmSpec, make_spec
from .registry import AlgorithmRegistry
from .trainer import Trainer
from .base import BackendBase

__all__ = [
    "AlgorithmSpec",
    "make_spec",
    "AlgorithmRegistry",
    "Trainer",
    "BackendBase",
]