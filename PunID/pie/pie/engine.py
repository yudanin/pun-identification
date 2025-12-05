"""
Main Pun Identification Engine for PIE

Orchestrates pun detection using:
- Claude LLM for pun analysis
- FrameNet for frame distance (no LLM fallback)
- Validators for verification
"""

import json
import logging
import re
from typing import Optional
import nltk

try:
    nltk.data.find('corpora/framenet_v17')
except LookupError:
    nltk.download('framenet_v17', quiet=True)

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
4. An explanation of how the pun works

Respond in this exact JSON format:
{
  "has_pun": 1 or 0,
  "puns": [
    {
      "word_or_expression": "the pun word/phrase",
      "pun_type": "HOMOPHONIC|HOMOGRAPHIC|RECURSIVE|ANTANACLASIS",
      "sense1": "first meaning",
      "sense2": "second meaning", 
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
            api_key_file: Optional[str] = "claudeapikey",
            model: str = "claude-sonnet-4-20250514",
            validate: bool = True
    ):
        """
        Initialize the Pun Identification Engine.

        Args:
            api_key: Anthropic API key. If None, will try to read from file or env var.
            api_key_file: Path to file containing API key (default: 'claudeapikey').
            model: Claude model to use for analysis.
            validate: Whether to run validation tests on identified puns.
        """
        self._model = model
        self._validate = validate
        self._client = None
        self._framenet_service = FrameNetService()
        self._validator = PunValidator()

        # Try to get API key from: 1) parameter, 2) file, 3) environment variable
        resolved_api_key = api_key

        if not resolved_api_key and api_key_file:
            resolved_api_key = self._read_api_key_from_file(api_key_file)

        if not resolved_api_key:
            import os
            resolved_api_key = os.environ.get('ANTHROPIC_API_KEY')

        if resolved_api_key:
            self._init_client(resolved_api_key)

    def _read_api_key_from_file(self, filepath: str) -> Optional[str]:
        """Read API key from a file in the project root."""
        import os

        path = os.path.join(os.path.dirname(__file__), '..', '..', filepath)
        try:
            with open(path, 'r') as f:
                return f.read().strip()
        except Exception as e:
            logger.debug(f"Could not read API key from {path}: {e}")
            return None

    def _init_client(self, api_key: str):
        """Initialize the Anthropic client."""
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
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

        # Step 3: Build pun instances with frame information from FrameNet
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
            # Get pun type - keep raw string, try to match enum
            pun_type_str = pun_data.get("pun_type", "unknown")
            try:
                pun_type_enum = PunType[pun_type_str.upper()]
            except KeyError:
                pun_type_enum = None

            # Get the pun word and senses
            pun_word = pun_data.get("word_or_expression", "")
            sense1 = pun_data.get("sense1", "")
            sense2 = pun_data.get("sense2", "")

            # Look up frames for this word in FrameNet
            frame_distance = self._get_frame_distance_for_word(pun_word, sense1, sense2)

            pun = PunInstance(
                word_or_expression=pun_word,
                pun_type=pun_type_str,
                pun_type_enum=pun_type_enum,
                sense1=sense1,
                sense2=sense2,
                frame_distance=frame_distance,
                explanation=pun_data.get("explanation", ""),
                context_words=[],
                validation=None,
                confidence=0.0
            )
            puns.append(pun)

        return puns

    def _get_frame_distance_for_word(
            self,
            word: str,
            sense1: str,
            sense2: str
    ) -> FrameDistance:
        """
        Look up frames for a word in FrameNet and calculate distance.
        """
        # Get all frames for this word from FrameNet
        frames = self._framenet_service.get_frames_for_word(word)

        if len(frames) == 0:
            return FrameDistance(
                sense1_frame=None,
                sense2_frame=None,
                distance=-1,
                distance_type="no_frames",
                explanation=f"No frames found for '{word}' in FrameNet"
            )

        if len(frames) == 1:
            return FrameDistance(
                sense1_frame=frames[0],
                sense2_frame=None,
                distance=-1,
                distance_type="insufficient_frames",
                explanation=f"Only 1 frame found for '{word}' in FrameNet: {frames[0].frame_name}"
            )

        # We have 2+ frames - use first two for now
        # TODO: Could improve by matching frames to senses
        frame1 = frames[0]
        frame2 = frames[1]

        # Calculate distance between these frames
        frame_distance = self._framenet_service.calculate_frame_distance(
            frame1.frame_name, frame2.frame_name
        )

        # Override with actual frame objects
        frame_distance.sense1_frame = frame1
        frame_distance.sense2_frame = frame2

        return frame_distance

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