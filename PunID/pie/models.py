"""
Data models for PIE - Pun Identification Engine
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PunType(str, Enum):
    """Types of puns supported by PIE."""

    HOMOPHONIC = "homophonic"
    """Similar-sounding words. Example: 'atheism is a non-prophet institution'"""

    HOMOGRAPHIC = "homographic"
    """Same spelling, different meanings. Example: 'my shoe is a foot long'"""

    RECURSIVE = "recursive"
    """Term-dependent / self-referential twists. Example: 'Immanuel doesn't pun, he Kant'"""

    ANTANACLASIS = "antanaclasis"
    """Same word repeated with different senses. 
    Example: 'We must all hang together or we shall all hang separately.'"""


@dataclass
class FrameInfo:
    """FrameNet frame information for a word sense."""

    frame_name: str
    """Name of the FrameNet frame."""

    frame_definition: str
    """Definition of the frame."""

    lexical_unit: str
    """The lexical unit (word.POS) in this frame."""

    frame_elements: list[str] = field(default_factory=list)
    """Core frame elements."""


@dataclass
class FrameDistance:
    """Frame distance measurement between two senses."""

    sense1_frame: Optional[FrameInfo]
    """Frame for the first sense."""

    sense2_frame: Optional[FrameInfo]
    """Frame for the second sense."""

    distance: float
    """Numeric distance between frames (0 = same frame, higher = more distant)."""

    distance_type: str
    """How distance was calculated: 'graph', 'semantic', 'estimated'."""

    explanation: str
    """Human-readable explanation of the frame distance."""


@dataclass
class ValidationResult:
    """Results from pun validation tests."""

    distributional_valid: bool
    """Whether distributional semantics validates both senses are plausibly activated."""

    distributional_explanation: str
    """Explanation of distributional validation."""

    substitution_valid: bool
    """Whether substitution test passes (one sense can replace another grammatically)."""

    substitution_explanation: str
    """Explanation of substitution test."""

    overall_confidence: float
    """Overall confidence score (0.0 to 1.0) that this is a valid pun."""


@dataclass
class PunInstance:
    """A single pun identified in the text."""

    word_or_expression: str
    """The word or expression that creates the pun."""

    pun_type: str
    """Classification of the pun type (raw string from analysis)."""

    pun_type_enum: Optional[PunType] = None
    """Matched PunType enum if known, None otherwise."""

    sense1: str = ""
    """First meaning/sense activated by the pun."""

    sense2: str = ""
    """Second meaning/sense activated by the pun."""

    frame_distance: Optional[FrameDistance] = None
    """Frame distance information between the two senses."""

    explanation: str = ""
    """Human-readable explanation of how the pun works."""

    context_words: list[str] = field(default_factory=list)
    """Words in the sentence that activate or support each sense."""

    validation: Optional[ValidationResult] = None
    """Results from validation tests."""

    confidence: float = 0.0
    """Confidence score for this pun identification (0.0 to 1.0)."""


@dataclass
class PunAnalysisResult:
    """Complete result of pun analysis on a sentence."""

    sentence: str
    """The original sentence analyzed."""

    has_pun: int
    """1 if sentence contains one or more puns, 0 otherwise."""

    puns: list[PunInstance] = field(default_factory=list)
    """List of identified puns."""

    analysis_notes: str = ""
    """Additional notes from the analysis."""

    raw_llm_response: str = ""
    """Raw response from Claude for debugging/audit."""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "sentence": self.sentence,
            "has_pun": self.has_pun,
            "puns": [
                {
                    "word_or_expression": p.word_or_expression,
                    "pun_type": p.pun_type,
                    "pun_type_enum": p.pun_type_enum.value if p.pun_type_enum else None,
                    "sense1": p.sense1,
                    "sense2": p.sense2,
                    "frame_distance": {
                        "distance": p.frame_distance.distance,
                        "distance_type": p.frame_distance.distance_type,
                        "explanation": p.frame_distance.explanation,
                        "sense1_frame": {
                            "frame_name": p.frame_distance.sense1_frame.frame_name,
                            "frame_definition": p.frame_distance.sense1_frame.frame_definition,
                        } if p.frame_distance.sense1_frame else None,
                        "sense2_frame": {
                            "frame_name": p.frame_distance.sense2_frame.frame_name,
                            "frame_definition": p.frame_distance.sense2_frame.frame_definition,
                        } if p.frame_distance.sense2_frame else None,
                    } if p.frame_distance else None,
                    "explanation": p.explanation,
                    "context_words": p.context_words,
                    "validation": {
                        "distributional_valid": p.validation.distributional_valid,
                        "distributional_explanation": p.validation.distributional_explanation,
                        "substitution_valid": p.validation.substitution_valid,
                        "substitution_explanation": p.validation.substitution_explanation,
                        "overall_confidence": p.validation.overall_confidence,
                    } if p.validation else None,
                    "confidence": p.confidence,
                }
                for p in self.puns
            ],
            "analysis_notes": self.analysis_notes,
        }