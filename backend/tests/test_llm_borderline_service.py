"""
Tests for LLM Borderline Assessment Service with Anti-Hallucination Guardrails.
"""
import pytest
import asyncio
import json
from unittest.mock import patch, AsyncMock, MagicMock
from app.services.llm_borderline_service import LLMBorderlineAssessor


class TestLLMBorderlineAssessor:
    """Test suite for LLM borderline assessor."""

    @pytest.fixture
    def assessor(self):
        """Create a test assessor instance."""
        return LLMBorderlineAssessor(
            model='gpt-4o-mini',
            distance_range=(0.27, 0.57),
            adjustment_strength=0.15,
            openai_api_key='test-key-123',
            min_confidence=0.60,
            allow_unknown=True
        )

    def test_initialization(self, assessor):
        """Test assessor initialization."""
        assert assessor.model == 'gpt-4o-mini'
        assert assessor.distance_low == 0.27
        assert assessor.distance_high == 0.57
        assert assessor.adjustment_strength == 0.15
        assert assessor.min_confidence == 0.60
        assert assessor.allow_unknown is True
        assert isinstance(assessor.cache, dict)

    def test_initialization_without_api_key(self):
        """Test that initialization fails without API key."""
        # Mock settings to have empty API key
        with patch('app.services.llm_borderline_service.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = ''
            with pytest.raises(ValueError, match="OpenAI API key required"):
                LLMBorderlineAssessor(
                    openai_api_key=None
                )

    def test_get_cache_key(self, assessor):
        """Test cache key generation (alphabetically sorted)."""
        key1 = assessor._get_cache_key("Apple", "Microsoft")
        key2 = assessor._get_cache_key("Microsoft", "Apple")

        assert key1 == key2
        assert key1 == ("Apple", "Microsoft")

    def test_adjust_similarity_same(self, assessor):
        """Test similarity adjustment for 'same' decision."""
        original = 0.65
        adjusted = assessor.adjust_similarity(original, 'same', 0.80)

        # Should increase: 0.65 + (0.15 * 1.0 * 0.80) = 0.77
        assert adjusted == pytest.approx(0.77, abs=0.01)
        assert 0.0 <= adjusted <= 1.0

    def test_adjust_similarity_different(self, assessor):
        """Test similarity adjustment for 'different' decision."""
        original = 0.65
        adjusted = assessor.adjust_similarity(original, 'different', 0.90)

        # Should decrease: 0.65 + (0.15 * -1.0 * 0.90) = 0.515
        assert adjusted == pytest.approx(0.515, abs=0.01)
        assert 0.0 <= adjusted <= 1.0

    def test_adjust_similarity_unknown(self, assessor):
        """Test similarity adjustment for 'unknown' decision."""
        original = 0.65
        adjusted = assessor.adjust_similarity(original, 'unknown', 0.80)

        # Should NOT change for 'unknown'
        assert adjusted == original

    def test_adjust_similarity_clamping(self, assessor):
        """Test that adjusted similarity is clamped to [0, 1]."""
        # Test upper bound
        adjusted_high = assessor.adjust_similarity(0.95, 'same', 1.0)
        assert adjusted_high <= 1.0

        # Test lower bound
        adjusted_low = assessor.adjust_similarity(0.10, 'different', 1.0)
        assert adjusted_low >= 0.0

    def test_validate_reasoning_too_short(self, assessor):
        """Test reasoning validation rejects short reasoning."""
        result = assessor._validate_reasoning(
            "Too short",
            "Apple Inc",
            "Apple Corp"
        )

        assert result is not None
        assert "too short" in result.lower()

    def test_validate_reasoning_generic_phrases(self, assessor):
        """Test reasoning validation detects generic phrases."""
        generic_phrases = [
            "I think they are the same company",
            "They look similar to me",
            "Based on my knowledge, they match",
            "Probably the same",
            "In my opinion they're different"
        ]

        for phrase in generic_phrases:
            result = assessor._validate_reasoning(
                phrase,
                "Apple Inc",
                "Apple Corp"
            )
            assert result is not None, f"Should reject: {phrase}"

    def test_validate_reasoning_valid(self, assessor):
        """Test reasoning validation accepts good reasoning."""
        valid_reasoning = (
            "Both names contain Microsoft as the core identifier. "
            "Corp and Corporation are synonymous corporate suffixes."
        )

        result = assessor._validate_reasoning(
            valid_reasoning,
            "Microsoft Corp",
            "Microsoft Corporation"
        )

        assert result is None  # No error

    def test_detect_hallucination(self, assessor):
        """Test hallucination detection for external knowledge use."""
        hallucinated_phrases = [
            "American Express is a financial services company",
            "Microsoft is known for Windows operating system",
            "Tesla operates in the automotive industry",
            "Amazon sells products online",
            "Apple is headquartered in Cupertino"
        ]

        for phrase in hallucinated_phrases:
            detected = assessor._detect_hallucination(phrase)
            assert detected is True, f"Should detect hallucination in: {phrase}"

    def test_detect_hallucination_valid(self, assessor):
        """Test hallucination detection doesn't flag valid reasoning."""
        valid_reasoning = (
            "Both names share the core word 'Express'. However, "
            "'American Express' and 'Express Scripts' have different "
            "secondary identifiers suggesting different entities."
        )

        detected = assessor._detect_hallucination(valid_reasoning)
        assert detected is False

    def test_create_unknown_response(self, assessor):
        """Test creation of unknown response."""
        response = assessor._create_unknown_response(
            "Company A",
            "Company B",
            0.65,
            guardrail_reason="Low confidence"
        )

        assert response['decision'] == 'unknown'
        assert response['llm_confidence'] == 0.0
        assert response['original_similarity'] == 0.65
        assert response['adjusted_similarity'] == 0.65  # No adjustment
        assert response['llm_reviewed'] is True
        assert response['guardrails_triggered'] is True
        assert response['guardrail_reason'] == "Low confidence"

    @pytest.mark.asyncio
    async def test_assess_single_pair_success(self, assessor):
        """Test successful single pair assessment."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            'decision': 'same',
            'confidence': 0.85,
            'reasoning': 'Both contain Microsoft as the core identifier with Corp and Corporation being synonymous suffixes'
        })

        with patch('openai.ChatCompletion.create', return_value=mock_response):
            result = await assessor._assess_single_pair(
                "Microsoft Corp",
                "Microsoft Corporation",
                0.70
            )

            assert result['decision'] == 'same'
            assert result['llm_confidence'] == 0.85
            assert result['llm_reviewed'] is True
            assert result['original_similarity'] == 0.70
            assert result['adjusted_similarity'] > 0.70  # Should be increased

    @pytest.mark.asyncio
    async def test_assess_single_pair_low_confidence_guardrail(self, assessor):
        """Test that low confidence decisions are converted to 'unknown'."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            'decision': 'same',
            'confidence': 0.45,  # Below threshold (0.60)
            'reasoning': 'Both names contain Apple but confidence is low due to different suffixes'
        })

        with patch('openai.ChatCompletion.create', return_value=mock_response):
            result = await assessor._assess_single_pair(
                "Apple Inc",
                "Apple Corp",
                0.65
            )

            # Should be converted to "unknown"
            assert result['decision'] == 'unknown'
            assert result['guardrails_triggered'] is True
            assert 'confidence' in result['guardrail_reason'].lower()

    @pytest.mark.asyncio
    async def test_assess_single_pair_weak_reasoning_guardrail(self, assessor):
        """Test that weak reasoning triggers guardrails."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            'decision': 'same',
            'confidence': 0.85,
            'reasoning': 'They look similar'  # Too vague
        })

        with patch('openai.ChatCompletion.create', return_value=mock_response):
            result = await assessor._assess_single_pair(
                "Apple",
                "Apple Inc",
                0.70
            )

            # Should be converted to "unknown" due to weak reasoning
            assert result['decision'] == 'unknown'
            assert result['guardrails_triggered'] is True

    @pytest.mark.asyncio
    async def test_assess_single_pair_hallucination_guardrail(self, assessor):
        """Test that hallucination is detected and triggers guardrail."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            'decision': 'different',
            'confidence': 0.90,
            'reasoning': 'American Express is a financial services company while American Airlines operates in aviation'
        })

        with patch('openai.ChatCompletion.create', return_value=mock_response):
            result = await assessor._assess_single_pair(
                "American Express",
                "American Airlines",
                0.60
            )

            # Should be flagged for hallucination
            assert result['decision'] == 'unknown'
            assert result['guardrails_triggered'] is True
            assert 'hallucination' in result['guardrail_reason'].lower()

    @pytest.mark.asyncio
    async def test_assess_single_pair_invalid_json(self, assessor):
        """Test handling of invalid JSON response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Not valid JSON"

        with patch('openai.ChatCompletion.create', return_value=mock_response):
            result = await assessor._assess_single_pair(
                "Company A",
                "Company B",
                0.65
            )

            # Should fall back to "unknown"
            assert result['decision'] == 'unknown'
            assert result['guardrails_triggered'] is True
            assert 'json' in result['guardrail_reason'].lower()

    @pytest.mark.asyncio
    async def test_assess_single_pair_api_error(self, assessor):
        """Test handling of API errors."""
        with patch('openai.ChatCompletion.create', side_effect=Exception("API Error")):
            result = await assessor._assess_single_pair(
                "Company A",
                "Company B",
                0.65
            )

            # Should fall back to "unknown"
            assert result['decision'] == 'unknown'
            assert result['guardrails_triggered'] is True

    @pytest.mark.asyncio
    async def test_assess_pairs_batch_with_cache(self, assessor):
        """Test batch assessment with caching."""
        pairs = [
            ("Google LLC", "Google Incorporated", 0.70),
            ("Amazon Web Services", "AWS", 0.75),
            ("Google LLC", "Google Incorporated", 0.70),  # Duplicate - should use cache
        ]

        # Create separate responses for each pair with proper reasoning
        google_response = MagicMock()
        google_response.choices = [MagicMock()]
        google_response.choices[0].message.content = json.dumps({
            'decision': 'same',
            'confidence': 0.85,
            'reasoning': 'Google LLC and Google Incorporated both contain Google as the primary identifier with LLC and Incorporated being different corporate structure suffixes for the same company'
        })

        amazon_response = MagicMock()
        amazon_response.choices = [MagicMock()]
        amazon_response.choices[0].message.content = json.dumps({
            'decision': 'same',
            'confidence': 0.88,
            'reasoning': 'Amazon Web Services is commonly abbreviated as AWS, both referring to the cloud computing division of Amazon'
        })

        call_count = 0
        responses = [google_response, amazon_response]

        def mock_create(*args, **kwargs):
            nonlocal call_count
            response = responses[min(call_count, len(responses) - 1)]
            call_count += 1
            return response

        with patch('openai.ChatCompletion.create', side_effect=mock_create):
            results = await assessor.assess_pairs_batch(pairs)

            # Cache should work: 2 unique pairs
            # Note: If guardrails trigger, there might be additional calls, but results should still be cached
            assert len(results) == 2, f"Expected 2 unique results, got {len(results)}"

            # Verify cache key exists
            cache_key = assessor._get_cache_key("Google LLC", "Google Incorporated")
            assert cache_key in results

            # Verify cache is populated (duplicate pair should use cache)
            assert len(assessor.cache) == 2, "Cache should contain 2 entries"

    @pytest.mark.asyncio
    async def test_assess_pairs_batch_progress_callback(self, assessor):
        """Test that progress callback is called."""
        pairs = [
            ("Company A", "Company B", 0.65),
            ("Company C", "Company D", 0.68),
        ]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            'decision': 'unknown',
            'confidence': 0.50,
            'reasoning': 'Insufficient information to determine if these are the same company'
        })

        progress_calls = []

        def progress_callback(current, total, phase):
            progress_calls.append((current, total, phase))

        with patch('openai.ChatCompletion.create', return_value=mock_response):
            await assessor.assess_pairs_batch(pairs, progress_callback=progress_callback)

            # Progress callback should have been called
            assert len(progress_calls) > 0

    def test_build_assessment_prompt(self, assessor):
        """Test prompt building."""
        prompt = assessor._build_assessment_prompt(
            "Microsoft Corp",
            "Microsoft Corporation",
            0.75
        )

        assert "Microsoft Corp" in prompt
        assert "Microsoft Corporation" in prompt
        assert "75.0%" in prompt
        assert "same company" in prompt.lower()
        assert "json" in prompt.lower()

    def test_get_system_prompt(self, assessor):
        """Test system prompt contains guardrail instructions."""
        system_prompt = assessor._get_system_prompt()

        # Check for key guardrail instructions
        assert "ONLY on the names provided" in system_prompt
        assert "honest about uncertainty" in system_prompt
        assert "unknown" in system_prompt
        assert "substantive reasoning" in system_prompt
        assert "Never guess" in system_prompt
        assert "external knowledge" in system_prompt or "external research" in system_prompt


class TestGuardrailScenarios:
    """Test comprehensive guardrail scenarios."""

    @pytest.fixture
    def assessor(self):
        """Create assessor with strict guardrails."""
        return LLMBorderlineAssessor(
            model='gpt-4o-mini',
            openai_api_key='test-key',
            min_confidence=0.70,  # Stricter threshold
            allow_unknown=True
        )

    @pytest.mark.asyncio
    async def test_borderline_case_with_phonetics(self, assessor):
        """Test borderline case that should benefit from LLM."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            'decision': 'different',
            'confidence': 0.92,
            'reasoning': 'American Express and American Airlines share the word American but have completely different business identifiers: Express versus Airlines, indicating different companies'
        })

        with patch('openai.ChatCompletion.create', return_value=mock_response):
            result = await assessor._assess_single_pair(
                "American Express",
                "American Airlines",
                0.68  # Borderline similarity
            )

            assert result['decision'] == 'different'
            assert result['llm_confidence'] == 0.92
            assert result['adjusted_similarity'] < 0.68  # Should be decreased
            assert result['guardrails_triggered'] is False  # Valid response

    @pytest.mark.asyncio
    async def test_abbreviation_case(self, assessor):
        """Test abbreviation matching."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            'decision': 'same',
            'confidence': 0.95,
            'reasoning': 'IBM is a well-known abbreviation for International Business Machines based on the initials of the three words'
        })

        with patch('openai.ChatCompletion.create', return_value=mock_response):
            result = await assessor._assess_single_pair(
                "IBM",
                "International Business Machines",
                0.55
            )

            assert result['decision'] == 'same'
            assert result['adjusted_similarity'] > 0.55

    @pytest.mark.asyncio
    async def test_ambiguous_case_should_be_unknown(self, assessor):
        """Test that truly ambiguous cases return 'unknown'."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            'decision': 'unknown',
            'confidence': 0.40,
            'reasoning': 'ABC Company versus ABC Corp - both use the ambiguous acronym ABC with generic corporate suffixes, insufficient distinguishing information to determine if same entity'
        })

        with patch('openai.ChatCompletion.create', return_value=mock_response):
            result = await assessor._assess_single_pair(
                "ABC Company",
                "ABC Corp",
                0.60
            )

            assert result['decision'] == 'unknown'
            assert result['adjusted_similarity'] == 0.60  # No change for unknown


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
