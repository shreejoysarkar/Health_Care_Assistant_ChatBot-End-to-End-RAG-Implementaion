"""
BGE-M3 Embedding Module
========================
Generates dense and sparse embeddings using the BAAI/bge-m3 model via FlagEmbedding.
Outputs are formatted for direct ingestion into Qdrant (dense vectors + sparse named vectors).

BGE-M3 natively supports three representation types:
  - Dense:   768-dim float vector (for ANN / cosine similarity)
  - Sparse:  token-weight dict  (for lexical / BM25-style matching)
  - ColBERT: multi-vector        (for late interaction re-ranking)

This module exposes dense + sparse for the Qdrant hybrid search pipeline,
with ColBERT available for optional re-ranking downstream.
"""

import os
import sys
from typing import List, Dict, Any, Optional

import numpy as np

# Ensure the project root is in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class BGEM3Embedder:
    """
    Wrapper around FlagEmbedding's BGEM3FlagModel for generating
    dense and sparse embeddings suitable for Qdrant hybrid search.

    Attributes:
        model_name: HuggingFace model identifier (default: BAAI/bge-m3).
        batch_size: Number of texts to embed per batch.
        max_length: Maximum token length for the encoder.
        use_fp16: Whether to use half-precision inference.
        model: The loaded BGEM3FlagModel instance.
        embedding_dim: Dimensionality of the dense embeddings (768 for bge-m3).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: int = 64,
        max_length: int = 8192,
        use_fp16: bool = True,
    ):
        """
        Initialize the BGE-M3 embedder.

        Args:
            model_name: HuggingFace model ID. Falls back to config's embedding_model.
            batch_size: Batch size for encoding.
            max_length: Maximum sequence length (bge-m3 supports up to 8192).
            use_fp16: Use FP16 for faster GPU inference.
        """
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.model = None
        self.embedding_dim = 1024  # bge-m3 dense dimension

        logger.info(
            f"BGEM3Embedder configured: model={self.model_name}, "
            f"batch_size={self.batch_size}, max_length={self.max_length}, fp16={self.use_fp16}"
        )

    def load_model(self) -> None:
        """Load the BGE-M3 model. Called lazily on first encode call."""
        if self.model is not None:
            return

        logger.info(f"Loading BGE-M3 model: {self.model_name} ...")
        try:
            from FlagEmbedding import BGEM3FlagModel

            self.model = BGEM3FlagModel(
                self.model_name,
                use_fp16=self.use_fp16,
            )
            # Verify embedding dimension from a test encode
            test_output = self.model.encode(
                ["test"], batch_size=1, max_length=32, return_dense=True, return_sparse=False, return_colbert_vecs=False
            )
            self.embedding_dim = test_output["dense_vecs"].shape[1]
            logger.info(f"BGE-M3 model loaded. Dense embedding dimension: {self.embedding_dim}")

        except ImportError:
            logger.error(
                "FlagEmbedding is not installed. Install with: pip install FlagEmbedding"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load BGE-M3 model: {e}")
            raise

    def encode(
        self,
        texts: List[str],
        return_sparse: bool = True,
        return_colbert: bool = False,
    ) -> Dict[str, Any]:
        """
        Encode a list of texts into dense (and optionally sparse/colbert) embeddings.

        Args:
            texts: List of text strings to embed.
            return_sparse: Whether to return sparse (lexical weight) vectors.
            return_colbert: Whether to return ColBERT multi-vectors.

        Returns:
            Dictionary with keys:
                - "dense": np.ndarray of shape (N, dim)
                - "sparse" (optional): list of dicts {token_id: weight}
                - "colbert" (optional): list of np.ndarray
        """
        self.load_model()

        if not texts:
            return {"dense": np.array([]), "sparse": [], "colbert": []}

        logger.info(f"Encoding {len(texts)} texts (sparse={return_sparse}, colbert={return_colbert}) ...")

        output = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert,
        )

        result: Dict[str, Any] = {
            "dense": output["dense_vecs"],
        }

        if return_sparse and "lexical_weights" in output:
            result["sparse"] = output["lexical_weights"]

        if return_colbert and "colbert_vecs" in output:
            result["colbert"] = output["colbert_vecs"]

        logger.info(
            f"Encoding complete. Dense shape: {result['dense'].shape}, "
            f"Sparse vectors: {len(result.get('sparse', []))}"
        )
        return result

    def encode_query(self, query: str) -> Dict[str, Any]:
        """
        Encode a single query string. Convenience wrapper around encode().

        Args:
            query: The search query text.

        Returns:
            Dictionary with 'dense' (1D array) and 'sparse' (single dict) keys.
        """
        result = self.encode([query], return_sparse=True, return_colbert=False)
        return {
            "dense": result["dense"][0],
            "sparse": result["sparse"][0] if result.get("sparse") else {},
        }

    @staticmethod
    def sparse_to_qdrant_format(sparse_dict: Dict) -> Dict[str, List]:
        """
        Convert a BGE-M3 sparse weight dict {token_id: weight} into
        Qdrant's named sparse vector format {"indices": [...], "values": [...]}.

        Args:
            sparse_dict: Dictionary mapping token IDs (int or str) to float weights.

        Returns:
            Qdrant-compatible sparse vector dict.
        """
        if not sparse_dict:
            return {"indices": [], "values": []}

        indices = []
        values = []
        for token_id, weight in sparse_dict.items():
            indices.append(int(token_id))
            values.append(float(weight))
        return {"indices": indices, "values": values}


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    embedder = BGEM3Embedder()

    # Test with a couple of medical sentences
    test_texts = [
        "Diabetes mellitus is a metabolic disorder characterized by chronic hyperglycemia.",
        "The treatment of acute myocardial infarction includes thrombolytic therapy.",
        "Pneumonia is an inflammatory condition of the lung primarily affecting the alveoli.",
    ]

    result = embedder.encode(test_texts, return_sparse=True)

    print(f"\n=== BGE-M3 Embedding Test ===")
    print(f"Dense embeddings shape: {result['dense'].shape}")
    print(f"Dense vector sample (first 5 dims): {result['dense'][0][:5]}")
    print(f"Number of sparse vectors: {len(result.get('sparse', []))}")

    if result.get("sparse"):
        sp = result["sparse"][0]
        print(f"Sparse vector 0 — non-zero entries: {len(sp)}")
        qdrant_sparse = BGEM3Embedder.sparse_to_qdrant_format(sp)
        print(f"Qdrant sparse format — indices: {qdrant_sparse['indices'][:5]}...")
        print(f"Qdrant sparse format — values:  {qdrant_sparse['values'][:5]}...")

    # Test query encoding
    query_result = embedder.encode_query("What are the symptoms of diabetes?")
    print(f"\nQuery dense shape: {query_result['dense'].shape}")
    print(f"Query sparse entries: {len(query_result['sparse'])}")
    print("===============================")
