"""
Qdrant Vector Store Indexing Module
====================================
Manages the full embedding-and-indexing pipeline:
  1. Loads chunked documents from JSON.
  2. Embeds chunks using BGE-M3 (dense + sparse).
  3. Creates / recreates a Qdrant collection with:
       - Dense vector config  (1024-dim, cosine distance)
       - Named sparse vector config ("bm25" — for lexical/sparse search)
       - Payload field indexes for metadata filtering
  4. Upserts points in batches with full metadata payloads.

Supports both Qdrant Cloud (via URL + API key) and local Qdrant instances.
"""

import os
import sys
import json
import uuid
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure the project root is in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    Distance,
    PointStruct,
    PayloadSchemaType,
)

from core.embedding import BGEM3Embedder
from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────
# Metadata field definitions for Qdrant payload indexing
# ──────────────────────────────────────────────────────────
# These fields will have indexes created in Qdrant so that
# filtered queries (e.g. chapter == "Cardiology") are fast.
INDEXED_METADATA_FIELDS: Dict[str, PayloadSchemaType] = {
    "chapter":            PayloadSchemaType.KEYWORD,
    "section":            PayloadSchemaType.KEYWORD,
    "subsection":         PayloadSchemaType.KEYWORD,
    "source":             PayloadSchemaType.KEYWORD,
    "chunk_type":         PayloadSchemaType.KEYWORD,
    "demographic_focus":  PayloadSchemaType.KEYWORD,
    "clinical_context":   PayloadSchemaType.KEYWORD,
    "token_count":        PayloadSchemaType.INTEGER,
}


class QdrantIndexer:
    """
    Orchestrates embedding generation and Qdrant vector store management.

    Workflow:
        indexer = QdrantIndexer()
        indexer.create_collection()           # create/recreate collection
        indexer.create_payload_indexes()      # index metadata fields
        indexer.index_chunks("Data/chunks/medical_data_chunks.json")  # embed & upsert
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        embedding_batch_size: int = 64,
        upsert_batch_size: int = 100,
    ):
        """
        Initialize the Qdrant indexer.

        Args:
            collection_name: Qdrant collection name. Falls back to config.
            qdrant_url: Qdrant server URL. Falls back to config.
            qdrant_api_key: Qdrant API key. Falls back to config.
            embedding_batch_size: Batch size for the BGE-M3 encoder.
            upsert_batch_size: Number of points per Qdrant upsert call.
        """
        settings = get_settings()

        self.collection_name = collection_name or settings.collection_name
        self.qdrant_url = qdrant_url or settings.qdrant_url
        self.qdrant_api_key = qdrant_api_key or settings.qdrant_api_key
        self.upsert_batch_size = upsert_batch_size

        # Initialize the embedder
        self.embedder = BGEM3Embedder(batch_size=embedding_batch_size)

        # Initialize Qdrant client
        self.client = self._create_client()

        logger.info(
            f"QdrantIndexer initialized: collection={self.collection_name}, "
            f"upsert_batch={self.upsert_batch_size}"
        )

    def _create_client(self) -> QdrantClient:
        """Create and return a QdrantClient based on available configuration."""
        if self.qdrant_url:
            logger.info(f"Connecting to Qdrant Cloud at {self.qdrant_url}")
            return QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                timeout=120,
            )
        else:
            # Fall back to local Qdrant instance (default localhost:6333)
            logger.info("Connecting to local Qdrant instance at localhost:6333")
            return QdrantClient(host="localhost", port=6333, timeout=120)

    # ──────────────────────────────────────────────────────
    # Collection Management
    # ──────────────────────────────────────────────────────

    def create_collection(self, recreate: bool = False) -> None:
        """
        Create the Qdrant collection with dense + sparse vector configuration.

        Args:
            recreate: If True, delete and recreate the collection even if it exists.
        """
        # Ensure the model is loaded so we know the embedding dimension
        self.embedder.load_model()
        dim = self.embedder.embedding_dim

        exists = self.client.collection_exists(self.collection_name)

        if exists and not recreate:
            logger.info(f"Collection '{self.collection_name}' already exists. Skipping creation.")
            return

        if exists and recreate:
            logger.warning(f"Deleting existing collection '{self.collection_name}' for recreation.")
            self.client.delete_collection(self.collection_name)

        logger.info(
            f"Creating collection '{self.collection_name}' with "
            f"dense_dim={dim} (cosine) + sparse vector 'bm25'"
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=dim,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "bm25": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )
        logger.info(f"Collection '{self.collection_name}' created successfully.")

    def create_payload_indexes(self) -> None:
        """
        Create payload field indexes on the collection for fast metadata filtering.
        This enables efficient filtered retrieval queries like:
            chapter == "Cardiology" AND clinical_context == "Emergency"
        """
        logger.info(f"Creating {len(INDEXED_METADATA_FIELDS)} payload indexes ...")
        for field_name, field_type in INDEXED_METADATA_FIELDS.items():
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
                logger.debug(f"  Indexed field: {field_name} ({field_type})")
            except Exception as e:
                # Index may already exist — that's fine
                logger.debug(f"  Index for '{field_name}' may already exist: {e}")
        logger.info("Payload index creation complete.")

    # ──────────────────────────────────────────────────────
    # Data Loading
    # ──────────────────────────────────────────────────────

    @staticmethod
    def load_chunks(chunks_path: str) -> List[Dict[str, Any]]:
        """
        Load chunked document data from a JSON file.

        Args:
            chunks_path: Path to the chunks JSON file.

        Returns:
            List of chunk dicts with 'text' and 'metadata' keys.
        """
        path = Path(chunks_path)
        if not path.exists():
            raise FileNotFoundError(f"Chunks file not found: {path}")

        logger.info(f"Loading chunks from {path} ...")
        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        logger.info(f"Loaded {len(chunks)} chunks.")
        return chunks

    # ──────────────────────────────────────────────────────
    # Embedding + Upsert Pipeline
    # ──────────────────────────────────────────────────────

    def index_chunks(
        self,
        chunks_path: str = "Data/chunks/medical_data_chunks.json",
        recreate_collection: bool = False,
    ) -> int:
        """
        Full pipeline: load chunks → embed → create collection → upsert to Qdrant.

        Args:
            chunks_path: Path to the chunks JSON produced by DocumentChunker.
            recreate_collection: Whether to recreate the collection from scratch.

        Returns:
            Total number of points upserted.
        """
        # 1. Load chunks
        chunks = self.load_chunks(chunks_path)
        if not chunks:
            logger.warning("No chunks to index.")
            return 0

        # 2. Create collection
        self.create_collection(recreate=recreate_collection)
        self.create_payload_indexes()

        # 3. Extract texts for embedding
        texts = [chunk["text"] for chunk in chunks]

        # 4. Encode all texts (batched internally by the embedder)
        logger.info(f"Generating embeddings for {len(texts)} chunks ...")
        t0 = time.time()
        embeddings = self.embedder.encode(texts, return_sparse=True, return_colbert=False)
        elapsed = time.time() - t0
        logger.info(f"Embedding generation complete in {elapsed:.1f}s")

        dense_vecs = embeddings["dense"]
        sparse_vecs = embeddings.get("sparse", [None] * len(texts))

        # 5. Build Qdrant points and upsert in batches
        total_upserted = 0
        total_batches = (len(chunks) + self.upsert_batch_size - 1) // self.upsert_batch_size

        logger.info(f"Upserting {len(chunks)} points in {total_batches} batches ...")

        for batch_idx in range(total_batches):
            start = batch_idx * self.upsert_batch_size
            end = min(start + self.upsert_batch_size, len(chunks))

            points = []
            for i in range(start, end):
                chunk = chunks[i]
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"chunk-{i}"))

                # Build the vector dict for this point
                vectors = {
                    "dense": dense_vecs[i].tolist(),
                }

                # Add sparse vector if available
                if sparse_vecs[i] is not None:
                    sparse_qdrant = BGEM3Embedder.sparse_to_qdrant_format(sparse_vecs[i])
                    vectors["bm25"] = sparse_qdrant

                # Build the payload (metadata + the original text for retrieval)
                payload = {
                    "text": chunk["text"],
                    "chunk_index": i,
                    **chunk.get("metadata", {}),
                }

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vectors,
                        payload=payload,
                    )
                )

            # Upsert the batch
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            total_upserted += len(points)

            if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
                logger.info(
                    f"  Batch {batch_idx + 1}/{total_batches} — "
                    f"{total_upserted}/{len(chunks)} points upserted"
                )

        logger.info(f"Indexing complete. Total points upserted: {total_upserted}")
        return total_upserted

    # ──────────────────────────────────────────────────────
    # Collection Info / Diagnostics
    # ──────────────────────────────────────────────────────

    def get_collection_info(self) -> Dict[str, Any]:
        """Return collection metadata and statistics."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": info.status.value,
            "config": str(info.config),
        }

    def verify_indexing(self, sample_size: int = 3) -> None:
        """Scroll a few points from the collection to verify indexing."""
        records = self.client.scroll(
            collection_name=self.collection_name,
            limit=sample_size,
            with_payload=True,
            with_vectors=False,
        )
        points = records[0]
        logger.info(f"\n=== Verification: {len(points)} sample points ===")
        for pt in points:
            payload = pt.payload
            logger.info(
                f"  ID: {pt.id} | Chapter: {payload.get('chapter')} | "
                f"Section: {payload.get('section')} | Type: {payload.get('chunk_type')} | "
                f"Tokens: {payload.get('token_count')}"
            )
            logger.info(f"    Text preview: {payload.get('text', '')[:120]}...")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    indexer = QdrantIndexer()

    # Run the full indexing pipeline
    total = indexer.index_chunks(
        chunks_path="Data/chunks/medical_data_chunks.json",
        recreate_collection=True,
    )

    # Print collection info
    info = indexer.get_collection_info()
    print(f"\n=== Collection Info ===")
    print(f"Name:          {info['name']}")
    print(f"Points Count:  {info['points_count']}")
    print(f"Vectors Count: {info['vectors_count']}")
    print(f"Status:        {info['status']}")
    print(f"========================")

    # Verify a few points
    indexer.verify_indexing(sample_size=3)
