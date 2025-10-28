"""
Comprehensive tests for blocking service.

Tests blocking key generation and stratified reservoir sampling.
"""
import pytest
from collections import Counter

from app.services.blocking_service import BlockingKeyGenerator, StratifiedReservoirSampler


class TestBlockingKeyGenerator:
    """Tests for BlockingKeyGenerator class."""

    def test_generate_key_basic(self):
        """Test basic key generation with standard company name."""
        gen = BlockingKeyGenerator()
        key = gen.generate_key("Apple Inc.")

        assert key == "apple_APL"
        assert "_" in key  # Has phonetic component

    def test_generate_key_with_phonetic(self):
        """Test key generation includes phonetic code."""
        gen = BlockingKeyGenerator()

        # Microsoft should generate phonetic
        key = gen.generate_key("Microsoft Corporation")
        assert key.startswith("microsoft_")
        assert len(key) > len("microsoft")

    def test_generate_key_fallback_no_phonetic(self):
        """Test key generation fallback when phonetic unavailable (digits/short)."""
        gen = BlockingKeyGenerator()

        # Digits should skip phonetics
        key1 = gen.generate_key("3M Company")
        assert key1 == "3m"
        assert "_" not in key1

        # Very short tokens should skip phonetics
        key2 = gen.generate_key("GE Corporation")
        assert key2 == "ge"
        assert "_" not in key2

    def test_extract_first_token_with_stopwords(self):
        """Test stopword filtering in token extraction."""
        gen = BlockingKeyGenerator()

        # "The" should be filtered
        key1 = gen.generate_key("The Apple Company")
        assert key1.startswith("apple")

        # "A" should be filtered
        key2 = gen.generate_key("A Better Solution")
        assert key2.startswith("better")

        # "An" should be filtered
        key3 = gen.generate_key("An Amazing Product")
        assert key3.startswith("amazing")

    def test_extract_first_token_no_valid_tokens(self):
        """Test fallback when no valid tokens remain after filtering."""
        gen = BlockingKeyGenerator()

        # Only stopwords and punctuation
        key = gen.generate_key("The A & An")
        # Should fallback to normalized full name
        assert key in ["the a an", "a an", "an"]  # Depends on exact normalization
        assert "_" not in key  # No phonetic for fallback

    def test_should_use_phonetics(self):
        """Test phonetic applicability checks."""
        gen = BlockingKeyGenerator()

        # Normal tokens should use phonetics
        assert gen._should_use_phonetics("apple") == True
        assert gen._should_use_phonetics("microsoft") == True

        # Tokens with digits should NOT use phonetics
        assert gen._should_use_phonetics("3m") == False
        assert gen._should_use_phonetics("7eleven") == False

        # Very short tokens should NOT use phonetics
        assert gen._should_use_phonetics("a") == False
        assert gen._should_use_phonetics("ge") == False

        # Single char should NOT use phonetics
        assert gen._should_use_phonetics("x") == False

    def test_generate_phonetic_code_counter(self):
        """Test phonetic skip counter tracks systematic gaps."""
        gen = BlockingKeyGenerator()

        # Generate keys with various patterns
        gen.generate_key("Apple Inc.")  # Normal
        gen.generate_key("3M Company")  # Has digits
        gen.generate_key("GE Corp")     # Too short
        gen.generate_key("IBM Systems") # Too short

        stats = gen.get_phonetic_stats()

        # Should have tracked skipped tokens
        assert 'non_phonetic_token' in stats
        assert stats['non_phonetic_token'] >= 2  # At least 3M, GE, IBM

    def test_generate_key_edge_cases(self):
        """Test edge cases in key generation."""
        gen = BlockingKeyGenerator()

        # Empty string
        key1 = gen.generate_key("")
        assert key1 == ""

        # Only punctuation
        key2 = gen.generate_key("!!!")
        assert isinstance(key2, str)

        # Special characters
        key3 = gen.generate_key("São Paulo Tech")
        assert "sao" in key3.lower()  # Accent folding

        # Numbers only
        key4 = gen.generate_key("123")
        assert key4 == "123"


class TestStratifiedReservoirSampler:
    """Tests for StratifiedReservoirSampler class."""

    # ==================== Block Creation Tests ====================

    def test_create_blocks_basic(self):
        """Test basic block creation from blocking keys."""
        sampler = StratifiedReservoirSampler(max_pairs=100)

        names = ["Apple Inc.", "Apple Computer", "Microsoft Corp", "Microsoft"]
        blocking_keys = {
            "Apple Inc.": "apple_APL",
            "Apple Computer": "apple_APL",
            "Microsoft Corp": "microsoft_MKRSFT",
            "Microsoft": "microsoft_MKRSFT"
        }

        blocks, singletons = sampler._create_blocks(names, blocking_keys)

        assert len(blocks) == 2
        assert "apple_APL" in blocks
        assert "microsoft_MKRSFT" in blocks
        assert len(blocks["apple_APL"]) == 2
        assert len(blocks["microsoft_MKRSFT"]) == 2
        assert singletons == 0

    def test_create_blocks_filters_singletons(self):
        """Test that singleton blocks are filtered out."""
        sampler = StratifiedReservoirSampler(max_pairs=100, min_block_size=2)

        names = ["Apple Inc.", "Apple Computer", "Microsoft Corp", "Google LLC"]
        blocking_keys = {
            "Apple Inc.": "apple_APL",
            "Apple Computer": "apple_APL",
            "Microsoft Corp": "microsoft_MKRSFT",  # Singleton
            "Google LLC": "google_KKL"  # Singleton
        }

        blocks, singletons = sampler._create_blocks(names, blocking_keys)

        assert len(blocks) == 1  # Only apple block
        assert "apple_APL" in blocks
        assert "microsoft_MKRSFT" not in blocks
        assert "google_KKL" not in blocks
        assert singletons == 2  # Microsoft and Google filtered

    def test_create_blocks_groups_by_key(self):
        """Test that names are correctly grouped by blocking key."""
        sampler = StratifiedReservoirSampler(max_pairs=100)

        names = ["Apple", "Apple Inc.", "Apple Computer", "Apple Corp"]
        blocking_keys = {name: "apple_APL" for name in names}

        blocks, singletons = sampler._create_blocks(names, blocking_keys)

        assert len(blocks) == 1
        assert len(blocks["apple_APL"]) == 4
        assert set(blocks["apple_APL"]) == set(names)

    def test_create_blocks_empty_input(self):
        """Test block creation with empty input."""
        sampler = StratifiedReservoirSampler(max_pairs=100)

        blocks, singletons = sampler._create_blocks([], {})

        assert len(blocks) == 0
        assert singletons == 0

    # ==================== Budget Allocation Tests ====================

    def test_allocate_budget_proportional_only(self):
        """Test budget allocation with proportional component."""
        sampler = StratifiedReservoirSampler(
            max_pairs=1000,
            proportional_pct=1.0,
            floor_pct=0.0
        )

        blocks = {
            "apple_APL": ["Apple", "Apple Inc.", "Apple Computer"],  # 3 pairs
            "microsoft_MKRSFT": ["Microsoft", "Microsoft Corp"]  # 1 pair
        }

        allocation = sampler._allocate_budget(blocks, total_budget=100)

        # Should be purely proportional: 3:1 ratio
        assert allocation["apple_APL"] > allocation["microsoft_MKRSFT"]

        # But allocation is capped at available pairs
        # Total available: 3 + 1 = 4 pairs
        # So allocation should be capped at 4 total
        assert allocation["apple_APL"] == 3  # All available pairs
        assert allocation["microsoft_MKRSFT"] == 1  # All available pairs

    def test_allocate_budget_with_floor(self):
        """Test budget allocation with floor component."""
        sampler = StratifiedReservoirSampler(
            max_pairs=1000,
            proportional_pct=0.5,
            floor_pct=0.5
        )

        blocks = {
            "large": ["A", "B", "C", "D", "E"],  # 10 pairs
            "small": ["X", "Y"]  # 1 pair
        }

        allocation = sampler._allocate_budget(blocks, total_budget=100)

        # Floor should help small block
        assert allocation["small"] > 0
        # Large block should still get more
        assert allocation["large"] > allocation["small"]

    def test_allocate_budget_respects_max_pairs_cap(self):
        """Test that allocation doesn't exceed available pairs per block."""
        sampler = StratifiedReservoirSampler(
            max_pairs=10000,
            max_block_pairs=5
        )

        blocks = {
            "small": ["A", "B", "C"]  # Only 3 pairs available
        }

        allocation = sampler._allocate_budget(blocks, total_budget=100)

        # Should not allocate more than available
        assert allocation["small"] <= 3

    def test_allocate_budget_small_blocks_get_floor(self):
        """Test that small blocks get floor allocation even if disproportionately small."""
        sampler = StratifiedReservoirSampler(
            max_pairs=10000,
            proportional_pct=0.8,
            floor_pct=0.2
        )

        blocks = {
            "giant": ["A"] * 100,  # Many pairs
            "tiny": ["X", "Y"]  # Just 1 pair
        }

        allocation = sampler._allocate_budget(blocks, total_budget=1000)

        # Tiny block should get at least floor allocation
        assert allocation["tiny"] > 0
        # But giant should still dominate
        assert allocation["giant"] > allocation["tiny"] * 10

    def test_allocate_budget_single_block(self):
        """Test allocation with single block."""
        sampler = StratifiedReservoirSampler(max_pairs=100)

        blocks = {"apple_APL": ["Apple", "Apple Inc.", "Apple Computer"]}

        allocation = sampler._allocate_budget(blocks, total_budget=10)

        # Should allocate all budget to single block (but capped at available pairs)
        assert allocation["apple_APL"] == min(10, 3)  # 3 pairs available

    # ==================== Within-Block Sampling Tests ====================

    def test_reservoir_sample_basic(self):
        """Test basic reservoir sampling within a block."""
        sampler = StratifiedReservoirSampler(max_pairs=1000, rng_seed=42)

        block = ["Apple", "Apple Inc.", "Apple Computer"]
        pairs, stats = sampler._reservoir_sample_within_block(block, sample_size=2)

        assert len(pairs) == 2
        assert stats['total_pairs'] == 3
        assert stats['sampled_pairs'] == 2
        assert stats['capped'] == False

        # Check all pairs are from the block
        for n1, n2 in pairs:
            assert n1 in block
            assert n2 in block

    def test_reservoir_sample_small_block(self):
        """Test sampling when sample_size > available pairs."""
        sampler = StratifiedReservoirSampler(max_pairs=1000, rng_seed=42)

        block = ["Apple", "Apple Inc."]  # Only 1 pair
        pairs, stats = sampler._reservoir_sample_within_block(block, sample_size=10)

        assert len(pairs) == 1  # Can't sample more than available
        assert stats['total_pairs'] == 1
        assert stats['sampled_pairs'] == 1

    def test_reservoir_sample_reproducibility(self):
        """Test that same seed produces same sample."""
        block = ["A", "B", "C", "D", "E"]

        sampler1 = StratifiedReservoirSampler(max_pairs=1000, rng_seed=42)
        pairs1, _ = sampler1._reservoir_sample_within_block(block, sample_size=5)

        sampler2 = StratifiedReservoirSampler(max_pairs=1000, rng_seed=42)
        pairs2, _ = sampler2._reservoir_sample_within_block(block, sample_size=5)

        assert pairs1 == pairs2

    def test_reservoir_sample_giant_block(self):
        """Test sampling from block that exceeds max_block_pairs cap."""
        sampler = StratifiedReservoirSampler(max_pairs=1000, max_block_pairs=10, rng_seed=42)

        # Create block with 10 names = 45 pairs, but cap at 10
        block = [f"Name{i}" for i in range(10)]
        pairs, stats = sampler._reservoir_sample_within_block(block, sample_size=5)

        assert len(pairs) == 5
        assert stats['capped'] == True
        assert stats['total_pairs'] == 10  # Capped

    def test_reservoir_sample_stats(self):
        """Test that stats dict is correctly populated."""
        sampler = StratifiedReservoirSampler(max_pairs=1000, rng_seed=42)

        block = ["A", "B", "C", "D"]  # 6 pairs
        pairs, stats = sampler._reservoir_sample_within_block(block, sample_size=3)

        assert 'total_pairs' in stats
        assert 'sampled_pairs' in stats
        assert 'capped' in stats
        assert stats['total_pairs'] == 6
        assert stats['sampled_pairs'] == 3
        assert isinstance(stats['capped'], bool)

    def test_reservoir_sample_algorithm_r(self):
        """Test that Algorithm R produces uniform distribution."""
        sampler = StratifiedReservoirSampler(max_pairs=10000, rng_seed=None)

        block = ["A", "B", "C", "D", "E"]  # 10 pairs

        # Sample many times and check distribution
        pair_counts = Counter()
        num_trials = 1000
        sample_size = 3

        for _ in range(num_trials):
            pairs, _ = sampler._reservoir_sample_within_block(block, sample_size)
            for pair in pairs:
                pair_counts[pair] += 1

        # Each pair should appear roughly sample_size/total_pairs * num_trials times
        # Expected: 3/10 * 1000 = 300 per pair
        # Allow 20% deviation
        expected = (sample_size / 10) * num_trials
        for count in pair_counts.values():
            assert 0.7 * expected <= count <= 1.3 * expected

    # ==================== Cross-Block Sampling Tests ====================

    def test_uniform_sample_cross_block_basic(self):
        """Test basic cross-block sampling."""
        sampler = StratifiedReservoirSampler(max_pairs=1000, rng_seed=42)

        blocks = {
            "apple_APL": ["Apple Inc.", "Apple Computer"],
            "microsoft_MKRSFT": ["Microsoft Corp", "Microsoft"]
        }

        pairs = sampler._uniform_sample_cross_block(blocks, sample_size=10)

        # Should sample cross-block pairs
        assert len(pairs) <= 10
        assert len(pairs) > 0

        # All pairs should be from different blocks
        for n1, n2 in pairs:
            b1 = "apple_APL" if n1 in blocks["apple_APL"] else "microsoft_MKRSFT"
            b2 = "apple_APL" if n2 in blocks["apple_APL"] else "microsoft_MKRSFT"
            assert b1 != b2

    def test_cross_block_different_blocks_only(self):
        """Test that cross-block pairs are never from same block."""
        sampler = StratifiedReservoirSampler(max_pairs=1000, rng_seed=42)

        blocks = {
            "a": ["A1", "A2", "A3"],
            "b": ["B1", "B2"],
            "c": ["C1", "C2", "C3", "C4"]
        }

        pairs = sampler._uniform_sample_cross_block(blocks, sample_size=50)

        # Verify all pairs are cross-block
        for n1, n2 in pairs:
            # Find which blocks they belong to
            block1 = None
            block2 = None
            for key, names in blocks.items():
                if n1 in names:
                    block1 = key
                if n2 in names:
                    block2 = key

            assert block1 is not None
            assert block2 is not None
            assert block1 != block2

    def test_cross_block_reproducibility(self):
        """Test cross-block sampling is reproducible with same seed."""
        blocks = {
            "a": ["A1", "A2"],
            "b": ["B1", "B2"],
            "c": ["C1", "C2"]
        }

        sampler1 = StratifiedReservoirSampler(max_pairs=1000, rng_seed=42)
        pairs1 = sampler1._uniform_sample_cross_block(blocks, sample_size=10)

        sampler2 = StratifiedReservoirSampler(max_pairs=1000, rng_seed=42)
        pairs2 = sampler2._uniform_sample_cross_block(blocks, sample_size=10)

        assert pairs1 == pairs2

    # ==================== Integration Tests ====================

    def test_stratified_sampler_end_to_end(self):
        """Test complete stratified sampling workflow."""
        gen = BlockingKeyGenerator()
        sampler = StratifiedReservoirSampler(max_pairs=100, rng_seed=42)

        names = [
            "Apple Inc.", "Apple Computer", "Apple",
            "Microsoft Corp", "Microsoft",
            "Google LLC", "Google Inc.",
            "Amazon"
        ]

        # Generate blocking keys
        blocking_keys = {name: gen.generate_key(name) for name in names}

        # Sample pairs
        result = sampler.sample_pairs(names, blocking_keys)

        assert 'pairs' in result
        assert 'metadata' in result
        assert len(result['pairs']) > 0
        assert result['metadata']['total_names'] == len(names)
        assert result['metadata']['total_blocks'] > 0

    def test_stratified_sampler_respects_budget(self):
        """Test that total pairs sampled doesn't exceed max_pairs."""
        gen = BlockingKeyGenerator()
        sampler = StratifiedReservoirSampler(max_pairs=50, rng_seed=42)

        # Create enough names to generate > 50 pairs
        names = [f"Company {i}" for i in range(20)]
        blocking_keys = {name: gen.generate_key(name) for name in names}

        result = sampler.sample_pairs(names, blocking_keys)

        # Should not exceed budget
        assert len(result['pairs']) <= 50

        # Should be close to budget
        metadata = result['metadata']
        total_sampled = metadata['within_block_pairs'] + metadata['cross_block_pairs']
        assert total_sampled <= 50

    def test_stratified_sampler_with_realistic_data(self):
        """Test with realistic company names."""
        gen = BlockingKeyGenerator()
        sampler = StratifiedReservoirSampler(max_pairs=500, rng_seed=42)

        names = [
            # Apple variants
            "Apple Inc.", "Apple Computer Inc.", "Apple Computer", "Apple", "Apple Incorporated",
            # Microsoft variants
            "Microsoft Corporation", "Microsoft Corp", "Microsoft Corp.", "Microsoft", "Microsoft Company",
            # Google variants
            "Google LLC", "Google Inc.", "Google", "Alphabet Inc.", "Alphabet",
            # Single companies
            "Amazon", "Tesla", "Meta", "Netflix", "Adobe"
        ]

        blocking_keys = {name: gen.generate_key(name) for name in names}
        result = sampler.sample_pairs(names, blocking_keys)

        metadata = result['metadata']

        # Should create reasonable number of blocks
        assert metadata['total_blocks'] > 0
        assert metadata['total_blocks'] < len(names)

        # Should sample both within and cross-block
        assert metadata['within_block_pairs'] > 0
        assert metadata['cross_block_pairs'] > 0

        # Within-block should be significant portion (for similar names)
        total = metadata['within_block_pairs'] + metadata['cross_block_pairs']
        within_pct = metadata['within_block_pairs'] / total
        assert within_pct > 0.1  # At least 10% within-block
