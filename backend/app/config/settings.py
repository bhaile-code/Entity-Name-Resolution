"""
Application configuration settings.
Centralizes all configuration in one place for easy modification.
"""
import os
from typing import List
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings with sensible defaults for prototype."""

    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8001"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"

    # API Configuration
    API_V1_PREFIX: str = "/api"
    PROJECT_NAME: str = "Company Name Standardizer"
    VERSION: str = "2.0.0"
    DESCRIPTION: str = "API for normalizing and grouping similar company names"

    # CORS Configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    # File Upload Configuration
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".csv"]

    # Matching Algorithm Configuration
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "85.0"))

    # ============================================================
    # EMBEDDING CONFIGURATION
    # ============================================================

    # OpenAI API Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_EMBEDDING_MODEL_SMALL: str = "text-embedding-3-small"
    OPENAI_EMBEDDING_MODEL_LARGE: str = "text-embedding-3-large"

    # Local Embedding Model
    LOCAL_EMBEDDING_MODEL: str = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Embedding dimensions (reduced from default 1536 for speed)
    EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "512"))

    # Similarity score weights (must sum to ~1.0)
    # These weights control how much each component contributes to the final similarity score
    WRATIO_WEIGHT: float = float(os.getenv("WRATIO_WEIGHT", "0.40"))  # Fuzzy string matching (typos)
    TOKEN_SET_WEIGHT: float = float(os.getenv("TOKEN_SET_WEIGHT", "0.15"))  # Token overlap (reduced from 0.40)
    EMBEDDING_WEIGHT: float = float(os.getenv("EMBEDDING_WEIGHT", "0.45"))  # Semantic similarity (NEW)

    # Default embedding mode: 'openai-small', 'openai-large', 'local', or 'disabled'
    DEFAULT_EMBEDDING_MODE: str = os.getenv("DEFAULT_EMBEDDING_MODE", "openai-small")

    # GMM Adaptive Threshold Configuration
    USE_ADAPTIVE_THRESHOLD: bool = os.getenv("USE_ADAPTIVE_THRESHOLD", "False").lower() == "true"
    GMM_MIN_SAMPLES: int = int(os.getenv("GMM_MIN_SAMPLES", "50"))
    GMM_MAX_PAIRS: int = int(os.getenv("GMM_MAX_PAIRS", "50000"))  # Cap pairwise collection for performance
    GMM_FALLBACK_T_HIGH: float = float(os.getenv("GMM_FALLBACK_T_HIGH", "92.0"))
    GMM_FALLBACK_T_LOW: float = float(os.getenv("GMM_FALLBACK_T_LOW", "80.0"))

    # ============================================================
    # HAC (Hierarchical Agglomerative Clustering) Configuration
    # ============================================================
    # Clustering mode: 'fixed', 'adaptive_gmm', or 'hac'
    # - fixed: Use fixed similarity threshold (fastest, deterministic)
    # - adaptive_gmm: Use GMM to calculate adaptive thresholds (data-driven, but unstable)
    # - hac: Use Hierarchical Agglomerative Clustering (deterministic, robust)
    CLUSTERING_MODE: str = os.getenv("CLUSTERING_MODE", "fixed")

    # HAC distance threshold (0-1 range, where distance = 1 - similarity)
    # Lower = stricter grouping (more clusters), Higher = looser grouping (fewer clusters)
    # Default 0.42 means similarity must be >= 0.58 (58%) to group
    # Common values:
    #   0.15 (85% similarity) - Very strict, minimal grouping
    #   0.42 (58% similarity) - Moderate (default)
    #   0.50 (50% similarity) - Permissive, more aggressive grouping
    HAC_DISTANCE_THRESHOLD: float = float(os.getenv("HAC_DISTANCE_THRESHOLD", "0.42"))

    # HAC linkage method: 'average', 'single', 'complete', 'ward'
    # - average: Uses average distance between all pairs (balanced, recommended)
    # - single: Uses minimum distance (creates long chains, sensitive to noise)
    # - complete: Uses maximum distance (creates compact clusters, conservative)
    # - ward: Minimizes variance (good for well-separated clusters)
    HAC_LINKAGE_METHOD: str = os.getenv("HAC_LINKAGE_METHOD", "average")

    # ============================================================
    # STRATIFIED SAMPLING CONFIGURATION
    # ============================================================

    # Blocking Configuration
    BLOCKING_MIN_BLOCK_SIZE: int = int(os.getenv("BLOCKING_MIN_BLOCK_SIZE", "2"))
    BLOCKING_MAX_BLOCK_PAIRS: int = int(os.getenv("BLOCKING_MAX_BLOCK_PAIRS", "5000"))

    # Sampling Budget Split: Within-Block vs Cross-Block
    SAMPLING_WITHIN_BLOCK_PCT: float = float(os.getenv("SAMPLING_WITHIN_BLOCK_PCT", "0.95"))
    SAMPLING_CROSS_BLOCK_PCT: float = float(os.getenv("SAMPLING_CROSS_BLOCK_PCT", "0.05"))

    # Within-Block Allocation: Proportional vs Floor
    SAMPLING_PROPORTIONAL_PCT: float = float(os.getenv("SAMPLING_PROPORTIONAL_PCT", "0.80"))
    SAMPLING_FLOOR_PCT: float = float(os.getenv("SAMPLING_FLOOR_PCT", "0.20"))

    # Reproducibility
    SAMPLING_RNG_SEED: int = int(os.getenv("SAMPLING_RNG_SEED", "42"))

    # Common corporate suffixes to normalize
    CORPORATE_SUFFIXES: List[str] = [
        "inc", "incorporated", "corp", "corporation", "ltd", "limited",
        "llc", "llp", "co", "company", "group", "holdings", "partners",
        "enterprises", "international", "global", "industries", "solutions",
        "plc", "ag", "sa", "nv", "bv", "gmbh", "spa"
    ]

    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: Path = Path("logs")
    LOG_FILE: str = "app.log"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @classmethod
    def get_log_path(cls) -> Path:
        """Get full path to log file, creating directory if needed."""
        cls.LOG_DIR.mkdir(exist_ok=True)
        return cls.LOG_DIR / cls.LOG_FILE

    @classmethod
    def validate_sampling_config(cls) -> None:
        """
        Validate stratified sampling configuration constraints.

        Ensures:
        - Within-block + cross-block percentages sum to 1.0
        - Proportional + floor percentages sum to 1.0
        - Min block size >= 2 (singletons should be skipped)
        - All percentages are in [0.0, 1.0] range

        Raises:
            AssertionError: If any validation constraint fails
        """
        # Check percentage sums
        within_cross_sum = cls.SAMPLING_WITHIN_BLOCK_PCT + cls.SAMPLING_CROSS_BLOCK_PCT
        prop_floor_sum = cls.SAMPLING_PROPORTIONAL_PCT + cls.SAMPLING_FLOOR_PCT

        assert abs(within_cross_sum - 1.0) < 1e-6, \
            f"SAMPLING_WITHIN_BLOCK_PCT + SAMPLING_CROSS_BLOCK_PCT must equal 1.0 (got {within_cross_sum:.4f})"

        assert abs(prop_floor_sum - 1.0) < 1e-6, \
            f"SAMPLING_PROPORTIONAL_PCT + SAMPLING_FLOOR_PCT must equal 1.0 (got {prop_floor_sum:.4f})"

        # Check percentage ranges
        assert 0.0 <= cls.SAMPLING_WITHIN_BLOCK_PCT <= 1.0, \
            f"SAMPLING_WITHIN_BLOCK_PCT must be in [0.0, 1.0] (got {cls.SAMPLING_WITHIN_BLOCK_PCT})"

        assert 0.0 <= cls.SAMPLING_CROSS_BLOCK_PCT <= 1.0, \
            f"SAMPLING_CROSS_BLOCK_PCT must be in [0.0, 1.0] (got {cls.SAMPLING_CROSS_BLOCK_PCT})"

        assert 0.0 <= cls.SAMPLING_PROPORTIONAL_PCT <= 1.0, \
            f"SAMPLING_PROPORTIONAL_PCT must be in [0.0, 1.0] (got {cls.SAMPLING_PROPORTIONAL_PCT})"

        assert 0.0 <= cls.SAMPLING_FLOOR_PCT <= 1.0, \
            f"SAMPLING_FLOOR_PCT must be in [0.0, 1.0] (got {cls.SAMPLING_FLOOR_PCT})"

        # Check block size constraints
        assert cls.BLOCKING_MIN_BLOCK_SIZE >= 2, \
            f"BLOCKING_MIN_BLOCK_SIZE must be >= 2 to skip singletons (got {cls.BLOCKING_MIN_BLOCK_SIZE})"

        assert cls.BLOCKING_MAX_BLOCK_PAIRS > 0, \
            f"BLOCKING_MAX_BLOCK_PAIRS must be > 0 (got {cls.BLOCKING_MAX_BLOCK_PAIRS})"

        # Check GMM_MAX_PAIRS is reasonable
        assert cls.GMM_MAX_PAIRS > 0, \
            f"GMM_MAX_PAIRS must be > 0 (got {cls.GMM_MAX_PAIRS})"


# Singleton instance
settings = Settings()

# Validate sampling configuration on module load
settings.validate_sampling_config()
