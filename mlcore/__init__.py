"""
mlcore
======
"""

from mlcore.core.registry import MODEL_REGISTRY

# force registration
import mlcore.classification
import mlcore.regression

__all__ = [
    "MODEL_REGISTRY",
]

__version__ = "0.1.0"