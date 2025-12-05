"""
Main Pun Identification Engine for PIE

Orchestrates pun detection using:
- Claude LLM for analysis
- FrameNet for frame distance
- Validators for verification
"""

import json
import logging
import re
from typing import Optional

from .models import (
    PunAnalysisResult, 
    PunInstance, 
    PunType, 
    FrameInfo, 
    FrameDistance
)
from .framenet_service import FrameNetService
from .validators import PunValidator

logger = logging.getLogger(__name__)


# System prompt for Claude pun analysis
PUN_ANALYSIS_PROMPT = """You are a linguistic expert specializing in pun identification and analysis.

Your task is to analyze a sentence and identify any puns it contains. For each pun found, provide:

1. The word or expression that creates the pun
2. The type of pun:
   - HOMOPHONIC: Similar-sounding words (e.g., "prophet" / "profit")
   - HOMOGRAPHIC: Same spelling, different meanings (e.g., "foot" as body part / unit of measurement)
   - RECURSIVE: Self-referential or term-dependent (e.g., "Immanuel doesn't pun, he Kant")
   - ANTANACLASIS: Same word repeated with different senses (e.g., "hang together" / "hang separately")

3. The two senses/meanings being played upon
4. FrameNet frames that best capture each sense (use standard FrameNet frame names if you know them)
5. An explanation of how the pun works

Respond in this exact JSON format:
{
  "has_pun": 1 or 0,
  "puns": [
    {
      "word_or_expression": "the pun word/phrase",
      "pun_type": "HOMOPHONIC|HOMOGRAPHIC|RECURSIVE|ANTANACLASIS",
      "sense1": "first meaning",
      "sense2": "second meaning", 
      "sense1_frame": "FrameNet frame name for sense 1",
      "sense2_frame": "FrameNet frame name for sense 2",
      "explanation": "explanation of how the pun works"
    }
  ],
  "analysis_notes": "any additional observations"
}

Be thorough but precise. Only identify genuine puns where multiple meanings are simultaneously activated.
If there is no pun, return {"has_pun": 0, "puns": [], "analysis_notes": "explanation of why no pun"}
"""


class PunIdentificationEngine:
    """
    Main engine for identifying and analyzing puns.
    
    Usage:
        engine = PunIdentificationEngine(api_key="your-anthropic-key")
        result = engine.analyze("Time flies like an arrow; fruit flies like a banana.")
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        validate: bool = True
    ):
        """
        Initialize the Pun Identification Engine.
        
        Args:
            api_key: Anthropic API key. If None, will look for ANTHROPIC_API_KEY env var.
            model: Claude model to use for analysis.
            validate: Whether to run validation tests on identified puns.
        """
        self._model = model
        self._validate = validate
        self._client = None
        self._framenet_service = FrameNetService()
        self._validator = PunValidator()
        
        if api_key:
            self._init_client(api_key)
    
    def _init_client(self, api_key: str):
        """Initialize the Anthropic client."""
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
            self._framenet_service.set_llm_client(self._client)
            self._validator.set_anthropic_client(self._client)
            logger.info("Anthropic client initialized")
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Anthropic client: {e}")
    
    def set_api_key(self, api_key: str):
        """Set or update the API key."""
        self._init_client(api_key)
    
    @property
    def is_configured(self) -> bool:
        """Check if the engine is properly configured."""
        return self._client is not None
    
    def analyze(self, sentence: str) -> PunAnalysisResult:
        """
        Analyze a sentence for puns.
        
        Args:
            sentence: The sentence to analyze.
            
        Returns:
            PunAnalysisResult containing the analysis.
        """
        if not self._client:
            raise RuntimeError("Engine not configured. Call set_api_key() first.")
        
        # Step 1: Get initial analysis from Claude
        raw_response = self._get_llm_analysis(sentence)
        
        # Step 2: Parse the response
        analysis_data = self._parse_llm_response(raw_response)
        
        # Step 3: Build pun instances with frame information
        puns = self._build_pun_instances(analysis_data, sentence)
        
        # Step 4: Validate each pun if enabled
        if self._validate:
            for pun in puns:
                validation = self._validator.validate_pun(sentence, pun)
                pun.validation = validation
                pun.confidence = validation.overall_confidence
        
        return PunAnalysisResult(
            sentence=sentence,
            has_pun=1 if puns else 0,
            puns=puns,
            analysis_notes=analysis_data.get("analysis_notes", ""),
            raw_llm_response=raw_response
        )
    
    def _get_llm_analysis(self, sentence: str) -> str:
        """Get pun analysis from Claude."""
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=2000,
                system=PUN_ANALYSIS_PROMPT,
                messages=[
                    {
                        "role": "user", 
                        "content": f"Analyze this sentence for puns:\n\n\"{sentence}\""
                    }
                ]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"LLM analysis error: {e}")
            raise RuntimeError(f"Failed to get LLM analysis: {e}")
    
    def _parse_llm_response(self, response: str) -> dict:
        """Parse the JSON response from Claude."""
        # Try to extract JSON from the response
        try:
            # First try direct parse
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON in the response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fallback: return empty analysis
        logger.warning(f"Could not parse LLM response as JSON: {response[:200]}...")
        return {"has_pun": 0, "puns": [], "analysis_notes": "Failed to parse response"}
    
    def _build_pun_instances(self, analysis_data: dict, sentence: str) -> list[PunInstance]:
        """Build PunInstance objects from analysis data."""
        puns = []
        
        for pun_data in analysis_data.get("puns", []):
            # Map pun type string to enum
            pun_type_str = pun_data.get("pun_type", "HOMOGRAPHIC").upper()
            try:
                pun_type = PunType[pun_type_str]
            except KeyError:
                pun_type = PunType.HOMOGRAPHIC
                logger.warning(f"Unknown pun type: {pun_type_str}")
            
            # Get frame information
            sense1_frame_name = pun_data.get("sense1_frame", "Unknown")
            sense2_frame_name = pun_data.get("sense2_frame", "Unknown")
            
            # Calculate frame distance
            frame_distance = self._calculate_frame_distance(
                sense1_frame_name,
                sense2_frame_name,
                pun_data.get("sense1", ""),
                pun_data.get("sense2", "")
            )
            
            pun = PunInstance(
                word_or_expression=pun_data.get("word_or_expression", ""),
                pun_type=pun_type,
                sense1=pun_data.get("sense1", ""),
                sense2=pun_data.get("sense2", ""),
                frame_distance=frame_distance,
                explanation=pun_data.get("explanation", ""),
                context_words=[],
                validation=None,
                confidence=0.0
            )
            puns.append(pun)
        
        return puns
    
    def _calculate_frame_distance(
        self,
        frame1_name: str,
        frame2_name: str,
        sense1_desc: str,
        sense2_desc: str
    ) -> FrameDistance:
        """
        Calculate frame distance between two senses.
        Uses FrameNet when available, falls back to LLM estimation.
        """
        # First try FrameNet service
        frame_distance = self._framenet_service.calculate_frame_distance(
            frame1_name, frame2_name
        )
        
        # If FrameNet couldn't calculate (distance < 0), use LLM estimation
        if frame_distance.distance < 0:
            frame_distance = self._estimate_frame_distance_via_llm(
                frame1_name, frame2_name, sense1_desc, sense2_desc
            )
        
        return frame_distance
    
    def _estimate_frame_distance_via_llm(
        self,
        frame1_name: str,
        frame2_name: str,
        sense1_desc: str,
        sense2_desc: str
    ) -> FrameDistance:
        """Estimate frame distance using Claude."""
        prompt = f"""Estimate the semantic frame distance between two word senses.

Frame 1: {frame1_name}
Sense 1: {sense1_desc}

Frame 2: {frame2_name}
Sense 2: {sense2_desc}

Using FrameNet's conceptual framework, estimate how semantically distant these frames are.
Consider:
- Are they in the same domain? (e.g., both about commerce, both about motion)
- Do they share frame elements?
- How many relation links would connect them in FrameNet's frame hierarchy?

Respond in this exact format:
DISTANCE: [number from 0-10, where 0=same frame, 10=maximally distant]
FRAME1_DEFINITION: [brief definition of frame 1]
FRAME2_DEFINITION: [brief definition of frame 2]
EXPLANATION: [explain the semantic distance between these frames]"""

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text
            
            # Parse distance
            distance = 5.0  # default
            distance_match = re.search(r'DISTANCE:\s*(\d+(?:\.\d+)?)', text)
            if distance_match:
                distance = float(distance_match.group(1))
            
            # Parse definitions
            frame1_def = ""
            frame2_def = ""
            def1_match = re.search(r'FRAME1_DEFINITION:\s*(.+?)(?=\n|FRAME2)', text, re.DOTALL)
            def2_match = re.search(r'FRAME2_DEFINITION:\s*(.+?)(?=\n|EXPLANATION)', text, re.DOTALL)
            if def1_match:
                frame1_def = def1_match.group(1).strip()
            if def2_match:
                frame2_def = def2_match.group(1).strip()
            
            # Parse explanation
            explanation = ""
            exp_match = re.search(r'EXPLANATION:\s*(.+)', text, re.DOTALL)
            if exp_match:
                explanation = exp_match.group(1).strip()
            
            # Create frame info objects
            frame1_info = FrameInfo(
                frame_name=frame1_name,
                frame_definition=frame1_def,
                lexical_unit="",
                frame_elements=[]
            )
            frame2_info = FrameInfo(
                frame_name=frame2_name,
                frame_definition=frame2_def,
                lexical_unit="",
                frame_elements=[]
            )
            
            # Cache the estimated frames
            self._framenet_service.cache_estimated_frame(frame1_info)
            self._framenet_service.cache_estimated_frame(frame2_info)
            
            return FrameDistance(
                sense1_frame=frame1_info,
                sense2_frame=frame2_info,
                distance=distance,
                distance_type="estimated",
                explanation=explanation
            )
            
        except Exception as e:
            logger.error(f"Frame distance estimation error: {e}")
            return FrameDistance(
                sense1_frame=None,
                sense2_frame=None,
                distance=5.0,
                distance_type="default",
                explanation=f"Could not estimate: {e}"
            )
    
    def analyze_batch(self, sentences: list[str]) -> list[PunAnalysisResult]:
        """
        Analyze multiple sentences for puns.
        
        Args:
            sentences: List of sentences to analyze.
            
        Returns:
            List of PunAnalysisResult objects.
        """
        return [self.analyze(sentence) for sentence in sentences]
    
    def get_status(self) -> dict:
        """Get engine status information."""
        return {
            "configured": self.is_configured,
            "model": self._model,
            "validation_enabled": self._validate,
            "framenet_nltk_available": self._framenet_service.using_nltk,
            "spacy_available": self._validator.spacy_available,
        }
