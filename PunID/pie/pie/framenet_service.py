"""
FrameNet integration for PIE - Pun Identification Engine

Provides frame distance calculations and frame information lookup.
Uses NLTK's FrameNet corpus when available, falls back to Claude-based estimation.
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
        except Exception as e:
            logger.error(f"Error getting frame '{frame_name}': {e}")
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
            f2 = self._fn.frame(frame2)
            
            # Check direct relations
            f1_relations = {rel.relatedFrame.name: rel.type.name for rel in f1.frameRelations}
            f2_relations = {rel.relatedFrame.name: rel.type.name for rel in f2.frameRelations}
            
            # Direct relation?
            if frame2 in f1_relations:
                rel_type = f1_relations[frame2]
                return (1.0, f"Direct relation: {frame1} --[{rel_type}]--> {frame2}")
            
            if frame1 in f2_relations:
                rel_type = f2_relations[frame1]
                return (1.0, f"Direct relation: {frame2} --[{rel_type}]--> {frame1}")
            
            # Common parent/child?
            common = set(f1_relations.keys()) & set(f2_relations.keys())
            if common:
                common_frame = list(common)[0]
                return (2.0, f"Share related frame: {common_frame}")
            
            # No direct connection found - estimate based on semantic distance
            return (5.0, f"No direct FrameNet relation found between {frame1} and {frame2}")
            
        except Exception as e:
            logger.error(f"Error calculating frame distance: {e}")
            return (-1, f"Error: {e}")


class EstimatedFrameNetProvider(FrameNetProvider):
    """
    Fallback provider that uses Claude to estimate frame information.
    Used when NLTK FrameNet is not available.
    """
    
    def __init__(self, llm_client=None):
        self._llm_client = llm_client
        self._frame_cache: dict[str, FrameInfo] = {}
    
    def set_llm_client(self, client):
        """Set the LLM client for frame estimation."""
        self._llm_client = client
    
    def get_frames_for_word(self, word: str, pos: Optional[str] = None) -> list[FrameInfo]:
        """Estimate frames for a word using common linguistic knowledge."""
        # This returns a placeholder - actual estimation happens in the engine
        # using the full context of the pun analysis
        return []
    
    def get_frame_by_name(self, frame_name: str) -> Optional[FrameInfo]:
        """Return cached frame or placeholder."""
        if frame_name in self._frame_cache:
            return self._frame_cache[frame_name]
        
        # Create a placeholder frame
        frame = FrameInfo(
            frame_name=frame_name,
            frame_definition=f"[Estimated frame: {frame_name}]",
            lexical_unit="",
            frame_elements=[]
        )
        self._frame_cache[frame_name] = frame
        return frame
    
    def calculate_frame_distance(
        self, 
        frame1: str, 
        frame2: str
    ) -> tuple[float, str]:
        """Return placeholder - actual calculation done by LLM in engine."""
        if frame1 == frame2:
            return (0.0, "Same frame")
        return (-1, "Requires LLM estimation")
    
    def cache_frame(self, frame_info: FrameInfo):
        """Cache a frame estimated by the LLM."""
        self._frame_cache[frame_info.frame_name] = frame_info


class FrameNetService:
    """
    High-level service for FrameNet operations.
    Automatically selects the best available provider.
    """
    
    def __init__(self, llm_client=None):
        self._nltk_provider = NLTKFrameNetProvider()
        self._estimated_provider = EstimatedFrameNetProvider(llm_client)
        
        if self._nltk_provider.is_available:
            self._primary_provider = self._nltk_provider
            logger.info("Using NLTK FrameNet provider")
        else:
            self._primary_provider = self._estimated_provider
            logger.info("Using estimated FrameNet provider (NLTK not available)")
    
    @property
    def using_nltk(self) -> bool:
        """Whether NLTK FrameNet is being used."""
        return self._nltk_provider.is_available
    
    def set_llm_client(self, client):
        """Set LLM client for estimation fallback."""
        self._estimated_provider.set_llm_client(client)
    
    def get_frames_for_word(self, word: str, pos: Optional[str] = None) -> list[FrameInfo]:
        """Get frames for a word, with fallback to estimation."""
        frames = self._primary_provider.get_frames_for_word(word, pos)
        if not frames and self._nltk_provider.is_available:
            # Try estimated provider as fallback
            frames = self._estimated_provider.get_frames_for_word(word, pos)
        return frames
    
    def get_frame_by_name(self, frame_name: str) -> Optional[FrameInfo]:
        """Get frame by name."""
        return self._primary_provider.get_frame_by_name(frame_name)
    
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
        
        distance, explanation = self._primary_provider.calculate_frame_distance(
            frame1_name, frame2_name
        )
        
        distance_type = "graph" if self._nltk_provider.is_available else "estimated"
        if distance < 0:
            distance_type = "unknown"
            distance = -1
        
        return FrameDistance(
            sense1_frame=frame1,
            sense2_frame=frame2,
            distance=distance,
            distance_type=distance_type,
            explanation=explanation
        )
    
    def cache_estimated_frame(self, frame_info: FrameInfo):
        """Cache a frame estimated by the LLM."""
        self._estimated_provider.cache_frame(frame_info)
