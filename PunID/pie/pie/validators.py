"""
Validation module for PIE - Pun Identification Engine

Implements:
- Distributional Semantics Validation
- Substitution Test
- Sense assignment validation via LLM
"""

import logging
from typing import Optional
from dataclasses import dataclass

from .models import ValidationResult, PunInstance

logger = logging.getLogger(__name__)


@dataclass
class DistributionalContext:
    """Context words associated with a particular sense."""
    sense: str
    context_words: list[str]
    association_strength: float  # 0.0 to 1.0


class SpaCyValidator:
    """
    Validator using spaCy for NLP analysis.
    Provides distributional semantics and grammatical validation.
    """
    
    def __init__(self):
        self._nlp = None
        self._available = False
        self._init_spacy()
    
    def _init_spacy(self):
        """Initialize spaCy."""
        try:
            import spacy
            # Try to load a model
            for model_name in ['en_core_web_lg', 'en_core_web_md', 'en_core_web_sm']:
                try:
                    self._nlp = spacy.load(model_name)
                    self._available = True
                    logger.info(f"spaCy initialized with model: {model_name}")
                    break
                except OSError:
                    continue
            
            if not self._available:
                logger.warning("No spaCy model available. Install with: python -m spacy download en_core_web_sm")
        except ImportError:
            logger.warning("spaCy not installed. Install with: pip install spacy")
            self._available = False
    
    @property
    def is_available(self) -> bool:
        return self._available
    
    def get_word_context(self, sentence: str, target_word: str) -> list[str]:
        """Extract context words around the target word."""
        if not self._available:
            return []
        
        doc = self._nlp(sentence)
        context_words = []
        
        for token in doc:
            if token.text.lower() == target_word.lower():
                # Get syntactic dependents and head
                context_words.extend([child.text for child in token.children])
                if token.head != token:
                    context_words.append(token.head.text)
                # Get nearby content words
                for other in doc:
                    if other != token and other.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']:
                        context_words.append(other.text)
        
        return list(set(context_words))
    
    def check_grammatical_substitution(
        self, 
        sentence: str, 
        original_word: str,
        substitute_word: str
    ) -> tuple[bool, str]:
        """
        Check if substituting one word for another maintains grammaticality.
        """
        if not self._available:
            return (True, "spaCy not available for grammatical check")
        
        # Parse original
        doc_original = self._nlp(sentence)
        
        # Find the target token
        target_token = None
        for token in doc_original:
            if token.text.lower() == original_word.lower():
                target_token = token
                break
        
        if not target_token:
            return (False, f"Could not find '{original_word}' in sentence")
        
        # Create substituted sentence
        substituted = sentence.replace(original_word, substitute_word)
        doc_substituted = self._nlp(substituted)
        
        # Basic grammaticality check: compare POS patterns
        original_pos = [t.pos_ for t in doc_original]
        substituted_pos = [t.pos_ for t in doc_substituted]
        
        if original_pos == substituted_pos:
            return (True, "POS pattern preserved after substitution")
        else:
            return (False, f"POS pattern changed: {original_pos} → {substituted_pos}")
    
    def get_word_vectors_similarity(self, word1: str, word2: str) -> float:
        """Get cosine similarity between word vectors."""
        if not self._available:
            return 0.0
        
        doc1 = self._nlp(word1)
        doc2 = self._nlp(word2)
        
        if doc1.vector_norm == 0 or doc2.vector_norm == 0:
            return 0.0
        
        return doc1.similarity(doc2)


class LLMValidator:
    """
    Validator using Claude LLM for semantic validation.
    Used as primary validator or fallback when spaCy is unavailable.
    """
    
    def __init__(self, anthropic_client=None):
        self._client = anthropic_client
    
    def set_client(self, client):
        """Set the Anthropic client."""
        self._client = client
    
    def validate_sense_activation(
        self,
        sentence: str,
        pun_word: str,
        sense1: str,
        sense2: str
    ) -> tuple[bool, str]:
        """
        Validate that both senses are plausibly activated in context.
        Uses distributional semantics reasoning.
        """
        if not self._client:
            return (True, "LLM validator not configured")
        
        prompt = f"""Analyze whether both meanings of a potential pun word are activated in this sentence.

Sentence: "{sentence}"
Pun word: "{pun_word}"
Sense 1: {sense1}
Sense 2: {sense2}

For each sense, identify:
1. What context words in the sentence support/activate this sense?
2. How strongly is this sense activated (weak/moderate/strong)?

Then determine: Are BOTH senses plausibly activated by the context? 
A true pun requires both meanings to be simultaneously accessible to the reader.

Respond in this exact format:
SENSE1_CONTEXT: [list context words]
SENSE1_STRENGTH: [weak/moderate/strong]
SENSE2_CONTEXT: [list context words]  
SENSE2_STRENGTH: [weak/moderate/strong]
BOTH_ACTIVATED: [yes/no]
EXPLANATION: [brief explanation]"""

        try:
            response = self._client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = response.content[0].text
            
            # Parse response
            both_activated = "BOTH_ACTIVATED: yes" in text.lower() or "both_activated: yes" in text
            
            # Extract explanation
            explanation = ""
            if "EXPLANATION:" in text:
                explanation = text.split("EXPLANATION:")[-1].strip()
            
            return (both_activated, explanation)
            
        except Exception as e:
            logger.error(f"LLM validation error: {e}")
            return (True, f"Validation error: {e}")
    
    def validate_substitution(
        self,
        sentence: str,
        pun_word: str,
        sense1: str,
        sense2: str
    ) -> tuple[bool, str]:
        """
        Perform substitution test: can we replace the pun word with 
        each sense's meaning and maintain coherence?
        """
        if not self._client:
            return (True, "LLM validator not configured")
        
        prompt = f"""Perform a substitution test for a potential pun.

Original sentence: "{sentence}"
Pun word: "{pun_word}"
Sense 1: {sense1}
Sense 2: {sense2}

Substitution test:
1. Replace "{pun_word}" with a word/phrase clearly meaning "{sense1}" - is the sentence grammatical and semantically coherent?
2. Replace "{pun_word}" with a word/phrase clearly meaning "{sense2}" - is the sentence grammatical and semantically coherent?

For a true pun, BOTH substitutions should produce grammatical sentences (even if the meanings differ).

Respond in this exact format:
SUBSTITUTION1: [the rewritten sentence with sense 1]
SUBSTITUTION1_VALID: [yes/no]
SUBSTITUTION2: [the rewritten sentence with sense 2]
SUBSTITUTION2_VALID: [yes/no]
TEST_PASSED: [yes/no]
EXPLANATION: [brief explanation]"""

        try:
            response = self._client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = response.content[0].text
            
            # Parse response
            test_passed = "TEST_PASSED: yes" in text.lower() or "test_passed: yes" in text
            
            # Extract explanation
            explanation = ""
            if "EXPLANATION:" in text:
                explanation = text.split("EXPLANATION:")[-1].strip()
            
            return (test_passed, explanation)
            
        except Exception as e:
            logger.error(f"LLM substitution test error: {e}")
            return (True, f"Validation error: {e}")


class PunValidator:
    """
    Main validation service combining spaCy and LLM validators.
    """
    
    def __init__(self, anthropic_client=None):
        self._spacy_validator = SpaCyValidator()
        self._llm_validator = LLMValidator(anthropic_client)
    
    def set_anthropic_client(self, client):
        """Set the Anthropic client for LLM validation."""
        self._llm_validator.set_client(client)
    
    @property
    def spacy_available(self) -> bool:
        return self._spacy_validator.is_available
    
    def validate_pun(
        self,
        sentence: str,
        pun: PunInstance
    ) -> ValidationResult:
        """
        Perform full validation of a pun instance.
        
        Uses both distributional semantics and substitution tests.
        """
        # Distributional semantics validation
        dist_valid, dist_explanation = self._llm_validator.validate_sense_activation(
            sentence=sentence,
            pun_word=pun.word_or_expression,
            sense1=pun.sense1,
            sense2=pun.sense2
        )
        
        # If spaCy is available, enhance with context words
        if self._spacy_validator.is_available:
            context_words = self._spacy_validator.get_word_context(
                sentence, pun.word_or_expression
            )
            if context_words:
                pun.context_words = context_words
        
        # Substitution test
        sub_valid, sub_explanation = self._llm_validator.validate_substitution(
            sentence=sentence,
            pun_word=pun.word_or_expression,
            sense1=pun.sense1,
            sense2=pun.sense2
        )
        
        # Calculate overall confidence
        confidence = 0.0
        if dist_valid:
            confidence += 0.5
        if sub_valid:
            confidence += 0.5
        
        # Adjust based on frame distance if available
        if pun.frame_distance and pun.frame_distance.distance > 0:
            # Higher frame distance = more "punny" = higher confidence
            frame_bonus = min(0.2, pun.frame_distance.distance * 0.04)
            confidence = min(1.0, confidence + frame_bonus)
        
        return ValidationResult(
            distributional_valid=dist_valid,
            distributional_explanation=dist_explanation,
            substitution_valid=sub_valid,
            substitution_explanation=sub_explanation,
            overall_confidence=confidence
        )
