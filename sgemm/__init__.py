import torch
from pathlib import Path

try:
    from . import _C
except ImportError as e:
    print(f"Error importing C++ extension '_C': {e}", flush=True)
    _C = None

from .sgemm_interface import (
    sgemm_rand,
)

__all__ = [
    "sgemm_rand",
]

__version__ = "0.0.1"
