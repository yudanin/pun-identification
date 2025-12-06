"""
FrameNet integration for PIE - Pun Identification Engine

Provides frame distance calculations and frame information lookup.
Uses NLTK's FrameNet corpus.
"""

import logging
from typing import Optional
from abc import ABC, abstractmethod

from .models import FrameInfo, FrameDistance

logger = logging.getLogger(__name__)


class FrameNetProvider(ABC):
    """Abstract base class for FrameNet data providers."""

    @abstractmethod
    def get_frames_for_word(self, word: str, pos: Optional[str] = None) -> list[FrameInfo]:
        """Get all frames that contain the given word as a lexical unit."""
        pass

    @abstractmethod
    def get_frame_by_name(self, frame_name: str) -> Optional[FrameInfo]:
        """Get frame information by frame name."""
        pass

    @abstractmethod
    def calculate_frame_distance(
            self,
            frame1: str,
            frame2: str
    ) -> tuple[float, str]:
        """
        Calculate semantic distance between two frames.

        Returns:
            Tuple of (distance, explanation)
            Distance is 0 for same frame, higher values indicate greater distance.
        """
        pass


class NLTKFrameNetProvider(FrameNetProvider):
    """FrameNet provider using NLTK's FrameNet corpus."""

    def __init__(self):
        self._fn = None
        self._available = False
        self._init_framenet()

    def _init_framenet(self):
        """Initialize NLTK FrameNet."""
        try:
            from nltk.corpus import framenet as fn
            # Test if data is available
            _ = fn.frames()
            self._fn = fn
            self._available = True
            logger.info("NLTK FrameNet initialized successfully")
        except LookupError:
            logger.warning("NLTK FrameNet data not downloaded. Run: nltk.download('framenet_v17')")
            self._available = False
        except Exception as e:
            logger.warning(f"Failed to initialize NLTK FrameNet: {e}")
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def get_frames_for_word(self, word: str, pos: Optional[str] = None) -> list[FrameInfo]:
        """Get frames containing the word as a lexical unit."""
        if not self._available:
            return []

        frames = []
        try:
            # Search for lexical units containing the word
            lus = self._fn.lus(r'(?i)^' + word + r'\.')
            for lu in lus:
                if pos and not lu.name.endswith('.' + pos):
                    continue
                frame = lu.frame
                frames.append(FrameInfo(
                    frame_name=frame.name,
                    frame_definition=frame.definition,
                    lexical_unit=lu.name,
                    frame_elements=[fe.name for fe in frame.FE.values() if fe.coreType == 'Core']
                ))
        except Exception as e:
            logger.error(f"Error looking up frames for '{word}': {e}")

        return frames

    def get_frame_by_name(self, frame_name: str) -> Optional[FrameInfo]:
        """Get frame by name."""
        if not self._available:
            return None

        try:
            frame = self._fn.frame(frame_name)
            return FrameInfo(
                frame_name=frame.name,
                frame_definition=frame.definition,
                lexical_unit="",
                frame_elements=[fe.name for fe in frame.FE.values() if fe.coreType == 'Core']
            )
        except Exception:
            # Frame not found - return None silently
            return None

    def calculate_frame_distance(
            self,
            frame1: str,
            frame2: str
    ) -> tuple[float, str]:
        """Calculate distance using frame relations."""
        if not self._available:
            return (-1, "FrameNet not available")

        if frame1 == frame2:
            return (0.0, "Same frame")

        try:
            f1 = self._fn.frame(frame1)
        except:
            return (-1, f"Frame not found: {frame1}")

        try:
            f2 = self._fn.frame(frame2)
        except:
            return (-1, f"Frame not found: {frame2}")

        try:
            # Check direct relations
            f1_relations = {}
            f2_relations = {}

            for rel in f1.frameRelations:
                try:
                    related = None
                    for attr in ['superFrameName', 'subFrameName', 'Parent', 'Child', 'relatedFrame']:
                        if hasattr(rel, attr):
                            val = getattr(rel, attr)
                            if val is not None:
                                related = val.name if hasattr(val, 'name') else str(val)
                                if related != frame1:
                                    break

                    if related and related != frame1:
                        rel_type = str(rel.type.name if hasattr(rel.type, 'name') else rel.type)
                        f1_relations[related] = rel_type
                except:
                    continue

            for rel in f2.frameRelations:
                try:
                    related = None
                    for attr in ['superFrameName', 'subFrameName', 'Parent', 'Child', 'relatedFrame']:
                        if hasattr(rel, attr):
                            val = getattr(rel, attr)
                            if val is not None:
                                related = val.name if hasattr(val, 'name') else str(val)
                                if related != frame2:
                                    break

                    if related and related != frame2:
                        rel_type = str(rel.type.name if hasattr(rel.type, 'name') else rel.type)
                        f2_relations[related] = rel_type
                except:
                    continue

            # Direct relation?
            if frame2 in f1_relations:
                return (1.0, f"Direct relation: {frame1} --[{f1_relations[frame2]}]--> {frame2}")

            if frame1 in f2_relations:
                return (1.0, f"Direct relation: {frame2} --[{f2_relations[frame1]}]--> {frame1}")

            # Common related frame?
            common = set(f1_relations.keys()) & set(f2_relations.keys())
            if common:
                return (2.0, f"Share related frame: {list(common)[0]}")

            return (5.0, f"No direct FrameNet relation found between {frame1} and {frame2}")

        except Exception as e:
            return (-1, f"Error: {e}")


class FrameNetService:
    """
    High-level service for FrameNet operations.
    """

    def __init__(self):
        self._nltk_provider = NLTKFrameNetProvider()

    @property
    def using_nltk(self) -> bool:
        """Whether NLTK FrameNet is being used."""
        return self._nltk_provider.is_available

    def get_frames_for_word(self, word: str, pos: Optional[str] = None) -> list[FrameInfo]:
        """Get frames for a word from FrameNet."""
        return self._nltk_provider.get_frames_for_word(word, pos)

    def get_frame_by_name(self, frame_name: str) -> Optional[FrameInfo]:
        """Get frame by name."""
        return self._nltk_provider.get_frame_by_name(frame_name)

    def calculate_frame_distance(
            self,
            frame1_name: str,
            frame2_name: str
    ) -> FrameDistance:
        """
        Calculate frame distance between two frames.

        Returns FrameDistance object with all details.
        """
        frame1 = self.get_frame_by_name(frame1_name)
        frame2 = self.get_frame_by_name(frame2_name)

        distance, explanation = self._nltk_provider.calculate_frame_distance(
            frame1_name, frame2_name
        )

        distance_type = "graph" if distance >= 0 else "unknown"

        return FrameDistance(
            sense1_frame=frame1,
            sense2_frame=frame2,
            distance=distance,
            distance_type=distance_type,
            explanation=explanation
        )