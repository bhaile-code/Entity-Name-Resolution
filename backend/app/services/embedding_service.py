"""
Embedding service for semantic similarity using OpenAI API or local models.
Supports three modes: OpenAI 3-large (best quality), OpenAI 3-small (balanced), and local (privacy).
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity

from app.utils.logger import setup_logger
from app.config import settings

logger = setup_logger(__name__)


class EmbeddingService:
    """
    Handles embedding generation for semantic similarity matching.

    Supports three modes:
    - 'openai-large': OpenAI text-embedding-3-large (best quality, ~90% accuracy)
    - 'openai-small': OpenAI text-embedding-3-small (balanced, ~85% accuracy)
    - 'local': sentence-transformers local model (privacy mode, ~75% accuracy)

    Features:
    - Intelligent caching to avoid re-computing embeddings
    - Graceful fallback when OpenAI API unavailable
    - Batch processing for efficiency
    """

    def __init__(self, mode: str = 'openai-small'):
        """
        Initialize the embedding service.

        Args:
            mode: One of 'openai-large', 'openai-small', or 'local'
        """
        self.mode = mode
        self.cache: Dict[str, np.ndarray] = {}
        self.openai_client = None
        self.local_model = None

        # Initialize based on mode
        if mode.startswith('openai'):
            self._init_openai()
        elif mode == 'local':
            self._init_local()
        else:
            raise ValueError(f"Invalid embedding mode: {mode}. Must be 'openai-large', 'openai-small', or 'local'")

        logger.info(f"Initialized EmbeddingService in mode: {mode}")

    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI

            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set in environment variables")

            self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info("OpenAI client initialized successfully")

        except ImportError:
            logger.error("OpenAI package not installed. Run: pip install openai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    def _init_local(self):
        """Initialize local sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading local model: {settings.LOCAL_EMBEDDING_MODEL} (this may take a moment...)")
            self.local_model = SentenceTransformer(settings.LOCAL_EMBEDDING_MODEL)
            logger.info(f"Local model loaded successfully ({settings.LOCAL_EMBEDDING_MODEL})")

        except ImportError:
            logger.error("sentence-transformers package not installed. Run: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise

    def get_model_name(self) -> str:
        """Get the current model name being used."""
        if self.mode == 'openai-large':
            return settings.OPENAI_EMBEDDING_MODEL_LARGE
        elif self.mode == 'openai-small':
            return settings.OPENAI_EMBEDDING_MODEL_SMALL
        elif self.mode == 'local':
            return settings.LOCAL_EMBEDDING_MODEL
        return "unknown"

    def get_embeddings_batch(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Get embeddings for a batch of texts.

        Uses caching to avoid re-computing embeddings for texts we've seen before.

        Args:
            texts: List of text strings to embed

        Returns:
            Dictionary mapping text -> embedding vector
        """
        # Check cache first
        uncached_texts = [t for t in texts if t not in self.cache]

        if not uncached_texts:
            logger.debug(f"All {len(texts)} embeddings found in cache")
            return {text: self.cache[text] for text in texts}

        logger.info(f"Computing embeddings for {len(uncached_texts)} texts ({len(texts) - len(uncached_texts)} cached)")

        # Generate embeddings for uncached texts
        if self.mode.startswith('openai'):
            new_embeddings = self._get_openai_embeddings_batch(uncached_texts)
        else:
            new_embeddings = self._get_local_embeddings_batch(uncached_texts)

        # Update cache
        self.cache.update(new_embeddings)

        # Return all embeddings (cached + new)
        return {text: self.cache[text] for text in texts}

    def _get_openai_embeddings_batch(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Get embeddings from OpenAI API."""
        try:
            model_name = self.get_model_name()

            # Call OpenAI API with batch
            response = self.openai_client.embeddings.create(
                model=model_name,
                input=texts,
                dimensions=settings.EMBEDDING_DIMENSIONS
            )

            # Extract embeddings
            embeddings = {}
            for text, embedding_obj in zip(texts, response.data):
                embeddings[text] = np.array(embedding_obj.embedding, dtype=np.float32)

            logger.info(f"Successfully retrieved {len(embeddings)} embeddings from OpenAI ({model_name})")
            return embeddings

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise RuntimeError(f"OpenAI API unavailable: {e}")

    def _get_local_embeddings_batch(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Get embeddings from local model."""
        try:
            # Encode batch
            embedding_vectors = self.local_model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True
            )

            # Build dictionary
            embeddings = {
                text: embedding.astype(np.float32)
                for text, embedding in zip(texts, embedding_vectors)
            }

            logger.info(f"Successfully generated {len(embeddings)} local embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"Local embedding generation failed: {e}")
            raise

    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Get embeddings (uses cache if available)
        embeddings = self.get_embeddings_batch([text1, text2])

        emb1 = embeddings[text1].reshape(1, -1)
        emb2 = embeddings[text2].reshape(1, -1)

        # Calculate cosine similarity
        sim = cosine_similarity(emb1, emb2)[0][0]

        # Ensure in [0, 1] range (cosine can be negative in theory)
        return float(max(0.0, min(1.0, sim)))

    def clear_cache(self):
        """Clear the embedding cache (useful for testing or memory management)."""
        cache_size = len(self.cache)
        self.cache.clear()
        logger.info(f"Cleared embedding cache ({cache_size} entries)")


def create_embedding_service(mode: str) -> Optional[EmbeddingService]:
    """
    Factory function to create an embedding service with error handling.

    Args:
        mode: Embedding mode ('openai-large', 'openai-small', 'local', or 'disabled')

    Returns:
        EmbeddingService instance, or None if mode is 'disabled' or creation fails
    """
    if mode == 'disabled':
        logger.info("Embeddings disabled")
        return None

    try:
        service = EmbeddingService(mode=mode)
        return service
    except Exception as e:
        logger.error(f"Failed to create embedding service in mode '{mode}': {e}")
        logger.warning("Falling back to disabled mode (fuzzy matching only)")
        return None
