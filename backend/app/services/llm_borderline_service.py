"""
LLM Borderline Assessment Service with Anti-Hallucination Guardrails.

Provides AI-powered assessment of ambiguous company name pairs (borderline similarity)
with robust guardrails to prevent hallucination and ensure honest "unknown" responses.
"""
import json
import asyncio
import logging
from typing import List, Tuple, Dict, Optional
from app.config import settings

logger = logging.getLogger(__name__)


class LLMBorderlineAssessor:
    """
    Assess borderline company name pairs using LLM with anti-hallucination guardrails.

    Guardrails:
    1. Explicit "unknown" option - LLM can abstain if uncertain
    2. Minimum confidence threshold (default: 0.60)
    3. Reasoning validation - reject generic/weak explanations
    4. Response format validation - enforce structured JSON
    5. Hallucination detection - flag external knowledge use
    6. No adjustment for "unknown" - preserve original scores
    """

    def __init__(
        self,
        model: str = 'gpt-4o-mini',
        distance_range: Tuple[float, float] = (0.27, 0.57),
        adjustment_strength: float = 0.15,
        openai_api_key: Optional[str] = None,
        min_confidence: float = 0.60,
        allow_unknown: bool = True
    ):
        """
        Initialize LLM borderline assessor with guardrails.

        Args:
            model: OpenAI model name (e.g., 'gpt-4o-mini')
            distance_range: (low, high) distance range for borderline pairs
            adjustment_strength: How much LLM affects similarity (0.0-1.0)
            openai_api_key: OpenAI API key (or from settings)
            min_confidence: Minimum confidence threshold (decisions below → "unknown")
            allow_unknown: Whether to allow "unknown" responses (strongly recommended: True)
        """
        self.model = model
        self.distance_low, self.distance_high = distance_range
        self.adjustment_strength = adjustment_strength
        self.min_confidence = min_confidence
        self.allow_unknown = allow_unknown
        self.cache = {}  # In-memory cache: (name1, name2) -> assessment

        # Initialize OpenAI API key
        try:
            self.openai_api_key = openai_api_key if openai_api_key is not None else settings.OPENAI_API_KEY
        except:
            self.openai_api_key = openai_api_key

        if not self.openai_api_key:
            raise ValueError("OpenAI API key required for LLM borderline assessment")

        logger.info(
            f"Initialized LLMBorderlineAssessor: model={model}, "
            f"distance_range={distance_range}, min_confidence={min_confidence}"
        )

    async def assess_pairs_batch(
        self,
        pairs: List[Tuple[str, str, float]],
        progress_callback: Optional[callable] = None
    ) -> Dict[Tuple[str, str], Dict]:
        """
        Assess multiple pairs in parallel batches with guardrails.

        Args:
            pairs: List of (name1, name2, current_similarity) tuples
            progress_callback: Optional callback(current, total, phase)

        Returns:
            Dict mapping (name1, name2) -> {
                'decision': 'same' | 'different' | 'unknown',
                'llm_confidence': 0.0-1.0,
                'reasoning': str,
                'original_similarity': float,
                'adjusted_similarity': float,
                'llm_reviewed': True,
                'guardrails_triggered': bool,
                'guardrail_reason': Optional[str]
            }
        """
        total_pairs = len(pairs)
        results = {}

        logger.info(f"Starting LLM batch assessment for {total_pairs} pairs")

        # Check cache first
        uncached_pairs = []
        for name1, name2, similarity in pairs:
            cache_key = self._get_cache_key(name1, name2)
            if cache_key in self.cache:
                results[cache_key] = self.cache[cache_key]
            else:
                uncached_pairs.append((name1, name2, similarity))

        logger.info(
            f"Cache hits: {len(results)}/{total_pairs}, "
            f"API calls needed: {len(uncached_pairs)}"
        )

        # Process uncached pairs in batches
        batch_size = settings.LLM_BORDERLINE_BATCH_SIZE
        for i in range(0, len(uncached_pairs), batch_size):
            batch = uncached_pairs[i:i + batch_size]

            # Call LLM for each pair in batch (parallel with asyncio.gather)
            tasks = [
                self._assess_single_pair(name1, name2, sim)
                for name1, name2, sim in batch
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Store results and cache
            for (name1, name2, sim), result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"LLM assessment failed for ({name1}, {name2}): {result}")
                    # Fallback: treat as "unknown"
                    result = self._create_unknown_response(
                        name1, name2, sim, error=str(result)
                    )

                cache_key = self._get_cache_key(name1, name2)
                results[cache_key] = result
                self.cache[cache_key] = result

            # Progress callback
            if progress_callback:
                completed = min(i + batch_size, len(uncached_pairs)) + len(self.cache) - len(uncached_pairs)
                progress_callback(completed, total_pairs, "llm_assessment")

        logger.info(f"LLM batch assessment complete: {len(results)} assessments")

        return results

    async def _assess_single_pair(
        self,
        name1: str,
        name2: str,
        current_similarity: float
    ) -> Dict:
        """
        Assess a single pair with guardrails.

        Returns assessment dict with guardrail metadata.
        """
        # Build prompt with anti-hallucination instructions
        prompt = self._build_assessment_prompt(name1, name2, current_similarity)

        # Call OpenAI API
        try:
            import openai
            openai.api_key = self.openai_api_key

            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=200
            )

            # Parse response
            content = response.choices[0].message.content

            # Try to extract JSON from response
            try:
                # Handle potential markdown code blocks
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                llm_response = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {content}")
                return self._create_unknown_response(
                    name1, name2, current_similarity,
                    guardrail_reason=f"Invalid JSON response: {str(e)}"
                )

            # Apply guardrails
            assessment = self._apply_guardrails(
                llm_response, name1, name2, current_similarity
            )

            return assessment

        except Exception as e:
            logger.error(f"LLM API error for ({name1}, {name2}): {e}", exc_info=True)
            return self._create_unknown_response(
                name1, name2, current_similarity, error=str(e)
            )

    def _get_system_prompt(self) -> str:
        """
        System prompt with strong anti-hallucination instructions.
        """
        return """You are an expert in company name matching. Your task is to determine if two company names refer to the same entity.

CRITICAL INSTRUCTIONS - ANTI-HALLUCINATION GUARDRAILS:

1. **Base your decision ONLY on the names provided**. Do NOT use external knowledge about companies.
   - ❌ WRONG: "American Express is a financial services company, so it's different from American Airlines"
   - ✅ CORRECT: "American Express and American Airlines share 'American' but have different core business identifiers"

2. **Be honest about uncertainty**. If you cannot confidently determine whether the names refer to the same company, respond with "unknown".
   - Use "unknown" when:
     - Names are ambiguous (e.g., "American" vs "American Co")
     - Insufficient distinguishing information
     - Confidence below 60%

3. **Provide substantive reasoning**. Your reasoning must:
   - Explain specific name similarities/differences
   - Reference actual words/patterns in the names
   - Avoid generic statements like "they look similar"

4. **Never guess or assume**. If uncertain, choose "unknown" rather than guessing.

5. **Response format** (JSON):
   {
     "decision": "same" | "different" | "unknown",
     "confidence": 0.0-1.0 (float, where 1.0 = absolutely certain),
     "reasoning": "Detailed explanation citing specific name elements"
   }

Examples of GOOD reasoning:
- "Both contain 'Microsoft' as the core identifier; 'Corp' and 'Corporation' are synonymous suffixes"
- "Different core words: 'Express' (delivery/finance) vs 'Airlines' (aviation), despite shared 'American'"
- "Insufficient information: 'ABC' could refer to multiple companies; confidence too low to determine"

Examples of BAD reasoning (will be rejected):
- "They look similar" (too vague)
- "I think they're the same company" (no evidence cited)
- "Based on my knowledge, American Express is a bank" (external knowledge not allowed)"""

    def _build_assessment_prompt(
        self,
        name1: str,
        name2: str,
        current_similarity: float
    ) -> str:
        """
        Build assessment prompt with context and guardrail reminders.
        """
        similarity_pct = current_similarity * 100

        return f"""Compare these two company names:

Company A: "{name1}"
Company B: "{name2}"

Current algorithmic similarity score: {similarity_pct:.1f}%

Question: Do these names refer to the SAME company?

Instructions:
- Analyze ONLY the names provided (no external research)
- Consider: exact matches, abbreviations, suffixes (Inc/Corp/Ltd), word order
- If uncertain or confidence <60%, respond "unknown"
- Provide detailed reasoning citing specific name elements

Respond in JSON format:
{{
  "decision": "same" | "different" | "unknown",
  "confidence": 0.0-1.0,
  "reasoning": "Your detailed explanation here"
}}"""

    def _apply_guardrails(
        self,
        llm_response: Dict,
        name1: str,
        name2: str,
        current_similarity: float
    ) -> Dict:
        """
        Apply guardrails to LLM response to prevent hallucination.

        Guardrails:
        1. Validate response structure
        2. Check confidence threshold
        3. Validate reasoning quality
        4. Detect generic/weak reasoning
        5. Detect hallucination (external knowledge use)
        """
        guardrails_triggered = False
        guardrail_reason = None

        # Guardrail 1: Validate response structure
        required_fields = ['decision', 'confidence', 'reasoning']
        if not all(field in llm_response for field in required_fields):
            guardrails_triggered = True
            guardrail_reason = "Invalid response structure: missing required fields"
            logger.warning(
                f"Guardrail triggered: {guardrail_reason} for ({name1}, {name2})"
            )
            return self._create_unknown_response(
                name1, name2, current_similarity,
                guardrail_reason=guardrail_reason
            )

        decision = llm_response['decision']
        confidence = llm_response['confidence']
        reasoning = llm_response['reasoning']

        # Guardrail 2: Validate decision value
        if decision not in ['same', 'different', 'unknown']:
            guardrails_triggered = True
            guardrail_reason = f"Invalid decision value: {decision}"
            logger.warning(
                f"Guardrail triggered: {guardrail_reason} for ({name1}, {name2})"
            )
            return self._create_unknown_response(
                name1, name2, current_similarity,
                guardrail_reason=guardrail_reason
            )

        # Guardrail 3: Check confidence threshold (only for same/different, not unknown)
        if decision != 'unknown' and confidence < self.min_confidence:
            guardrails_triggered = True
            guardrail_reason = (
                f"Confidence {confidence:.2f} below threshold {self.min_confidence}"
            )
            logger.info(
                f"Guardrail triggered: Low confidence for ({name1}, {name2}), "
                f"converting to 'unknown'"
            )
            decision = 'unknown'

        # Guardrail 4: Validate reasoning quality
        reasoning_issues = self._validate_reasoning(reasoning, name1, name2)
        if reasoning_issues:
            guardrails_triggered = True
            guardrail_reason = f"Weak reasoning: {reasoning_issues}"
            logger.warning(
                f"Guardrail triggered: {guardrail_reason} for ({name1}, {name2})"
            )
            decision = 'unknown'  # Downgrade to unknown if reasoning is weak

        # Guardrail 5: Detect hallucination indicators
        hallucination_detected = self._detect_hallucination(reasoning)
        if hallucination_detected:
            guardrails_triggered = True
            guardrail_reason = "Possible hallucination detected in reasoning"
            logger.warning(
                f"Guardrail triggered: {guardrail_reason} for ({name1}, {name2})"
            )
            decision = 'unknown'

        # Calculate adjusted similarity
        adjusted_similarity = self.adjust_similarity(
            current_similarity,
            decision,
            confidence
        )

        return {
            'decision': decision,
            'llm_confidence': confidence,
            'reasoning': reasoning,
            'original_similarity': current_similarity,
            'adjusted_similarity': adjusted_similarity,
            'llm_reviewed': True,
            'guardrails_triggered': guardrails_triggered,
            'guardrail_reason': guardrail_reason
        }

    def _validate_reasoning(
        self,
        reasoning: str,
        name1: str,
        name2: str
    ) -> Optional[str]:
        """
        Validate reasoning quality to prevent weak/generic explanations.

        Returns error message if reasoning is weak, None otherwise.
        """
        reasoning_lower = reasoning.lower()

        # Check minimum length
        if len(reasoning) < 30:
            return "Reasoning too short (< 30 chars)"

        # Detect generic phrases (hallucination indicators)
        generic_phrases = [
            "they look similar",
            "i think",
            "i believe",
            "probably",
            "might be",
            "could be",
            "seems like",
            "based on my knowledge",
            "as far as i know",
            "in my opinion"
        ]

        for phrase in generic_phrases:
            if phrase in reasoning_lower:
                return f"Generic/uncertain phrase detected: '{phrase}'"

        # Check if reasoning references the actual names
        # At least one word from each name should be mentioned (skip if names share words)
        name1_words = set(w.lower() for w in name1.split() if len(w) > 2)
        name2_words = set(w.lower() for w in name2.split() if len(w) > 2)
        reasoning_words = set(reasoning_lower.split())

        # If names share significant words, check for those shared words in reasoning
        shared_words = name1_words & name2_words
        if shared_words:
            # Check if at least one shared word is mentioned
            if not (shared_words & reasoning_words):
                return "Reasoning doesn't reference the actual names"
        else:
            # Check for at least 1 significant word overlap with each name
            overlap1 = name1_words & reasoning_words
            overlap2 = name2_words & reasoning_words

            if (not overlap1 and len(name1_words) > 0) or (not overlap2 and len(name2_words) > 0):
                return "Reasoning doesn't reference the actual names"

        return None  # Reasoning is valid

    def _detect_hallucination(self, reasoning: str) -> bool:
        """
        Detect potential hallucination (external knowledge use).

        Returns True if hallucination detected.
        """
        reasoning_lower = reasoning.lower()

        # Keywords indicating external knowledge use (not allowed)
        hallucination_indicators = [
            "is a company that",
            "is known for",
            "provides services",
            "sells products",
            "operates in",
            "headquartered in",
            "founded in",
            "ceo is",
            "specializes in",
            "industry leader",
            "market share",
            "financial services company",
            "technology company",
            "retail chain",
            "e-commerce",
            "software company"
        ]

        for indicator in hallucination_indicators:
            if indicator in reasoning_lower:
                logger.warning(
                    f"Hallucination detected: '{indicator}' found in reasoning"
                )
                return True

        return False

    def _create_unknown_response(
        self,
        name1: str,
        name2: str,
        current_similarity: float,
        error: Optional[str] = None,
        guardrail_reason: Optional[str] = None
    ) -> Dict:
        """
        Create an "unknown" response when LLM cannot provide confident answer.

        For "unknown" decisions, do NOT adjust similarity (keep original).
        """
        reasoning = guardrail_reason or error or "Unable to assess with confidence"

        return {
            'decision': 'unknown',
            'llm_confidence': 0.0,
            'reasoning': reasoning,
            'original_similarity': current_similarity,
            'adjusted_similarity': current_similarity,  # No adjustment for "unknown"
            'llm_reviewed': True,
            'guardrails_triggered': True,
            'guardrail_reason': guardrail_reason or error or "Unknown response created"
        }

    def adjust_similarity(
        self,
        original_similarity: float,
        llm_decision: str,
        llm_confidence: float
    ) -> float:
        """
        Adjust similarity based on LLM assessment with blending.

        Formula:
          adjusted = original + (strength × direction × confidence)

        Where:
          direction = +1 if "same", -1 if "different", 0 if "unknown"
          strength = self.adjustment_strength (e.g., 0.15)
          confidence = llm_confidence (0.0-1.0)

        "unknown" decisions do NOT adjust similarity (direction = 0).
        """
        if llm_decision == 'same':
            direction = 1.0
        elif llm_decision == 'different':
            direction = -1.0
        else:  # 'unknown'
            direction = 0.0  # No adjustment

        adjustment = self.adjustment_strength * direction * llm_confidence
        adjusted = original_similarity + adjustment

        # Clamp to [0, 1] range
        adjusted = max(0.0, min(1.0, adjusted))

        logger.debug(
            f"Similarity adjustment: {original_similarity:.3f} → {adjusted:.3f} "
            f"(decision={llm_decision}, confidence={llm_confidence:.2f})"
        )

        return adjusted

    def _get_cache_key(self, name1: str, name2: str) -> Tuple[str, str]:
        """Get normalized cache key (alphabetically sorted)."""
        return (name1, name2) if name1 < name2 else (name2, name1)
