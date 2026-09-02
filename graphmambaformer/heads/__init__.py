"""Output heads: mapping, adaptive routing, neural scoring, and the task heads."""

from .mapping_head import MappingHead
from .multitask_heads import MultiTaskHeads
from .router import ComplexityRouter
from .scoring_heads import ChainScoringHead, SeedScoringHead

__all__ = [
    "MappingHead",
    "ComplexityRouter",
    "SeedScoringHead",
    "ChainScoringHead",
    "MultiTaskHeads",
]
