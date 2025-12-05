"""
PIE - Pun Identification Engine

A system for identifying and analyzing puns using:
- Claude LLM for analysis and explanation
- FrameNet for frame distance calculations
- spaCy for NLP processing
- Distributional semantics for validation
"""

__version__ = "0.1.0"
__author__ = "PIE Development Team"

from .engine import PunIdentificationEngine
from .models import PunAnalysisResult, PunInstance, PunType

__all__ = [
    "PunIdentificationEngine",
    "PunAnalysisResult",
    "PunInstance",
    "PunType",
]
