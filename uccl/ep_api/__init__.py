"""Expert-parallel communication over the UCCL extension.

Import this module explicitly to load the CUDA/ROCm EP runtime. Importing
``uccl`` alone does not require PyTorch or initialize expert parallelism.
"""
from uccl.ep import Config, EventHandle
from .buffer import Buffer
from .utils import EventOverlap, check_nvlink_connections, initialize_uccl, destroy_uccl

__all__ = [
    "Config", "EventHandle", "Buffer", "EventOverlap",
    "check_nvlink_connections", "initialize_uccl", "destroy_uccl",
]
