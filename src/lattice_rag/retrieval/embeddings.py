from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result of cross-encoder reranking."""

    index: int
    score: float
    text: str


_ONNX_CACHE: dict[tuple[str, str, int], tuple[Any, Any]] = {}


class ONNXReranker:
    """CPU-quantized INT8 ONNX Reranker for bge-reranker-v2-m3."""

    def __init__(
        self,
        repo_id: str = "onnx-community/bge-reranker-v2-m3-ONNX",
        filename: str = "onnx/model_int8.onnx",
        max_length: int = 160,
    ) -> None:
        self.repo_id = repo_id
        self.filename = filename
        self.max_length = max_length
        self._session = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._session is None or self._tokenizer is None:
            cache_key = (self.repo_id, self.filename, self.max_length)
            if cache_key in _ONNX_CACHE:
                self._session, self._tokenizer = _ONNX_CACHE[cache_key]
                return

            import os
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer

            logger.info("Loading INT8 ONNX reranker from %s", self.repo_id)
            model_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.filename,
            )
            tok_path = hf_hub_download(
                repo_id=self.repo_id,
                filename="tokenizer.json",
            )
            opts = ort.SessionOptions()
            cpu_count = os.cpu_count() or 4
            opts.intra_op_num_threads = min(4, cpu_count)
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                model_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._tokenizer = Tokenizer.from_file(tok_path)
            self._tokenizer.enable_truncation(max_length=self.max_length)
            # Dynamic padding to batch max length for sub-100ms CPU inference
            self._tokenizer.enable_padding()
            _ONNX_CACHE[cache_key] = (self._session, self._tokenizer)

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Scores candidate documents against query using INT8 ONNX cross-encoder."""
        if not documents:
            return []

        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._session is not None

        pairs = [(query, doc) for doc in documents]
        encoded = self._tokenizer.encode_batch(pairs)

        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        outputs = self._session.run(
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        # Squeeze output logits into 1D float scores
        return outputs[0].flatten().astype(float).tolist()


class EmbeddingService:
    """FastEmbed wrapper for CPU-quantized dense embeddings and cross-encoder reranking."""

    def __init__(
        self,
        embed_model: str = "BAAI/bge-small-en-v1.5",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        embedder: Any = None,
        reranker: Any = None,
    ):
        """Initialize the embedding service with lazy-loaded models.

        Args:
            embed_model: Name of the text embedding model.
            reranker_model: Name of the cross-encoder model.
            embedder: Optional pre-initialized or mock embedder.
            reranker: Optional pre-initialized or mock reranker.
        """
        self.embed_model_name = embed_model
        self.reranker_model_name = reranker_model
        self._embed_model = embedder
        self._reranker_model = reranker

    @property
    def embedder(self) -> Any:
        """Lazy-load the text embedding model."""
        if self._embed_model is None:
            from fastembed import TextEmbedding

            logger.info("Loading embedding model %s", self.embed_model_name)
            self._embed_model = TextEmbedding(model_name=self.embed_model_name)
        return self._embed_model

    @property
    def reranker(self) -> Any:
        """Lazy-load the cross-encoder reranking model."""
        if self._reranker_model is None:
            if "v2-m3" in self.reranker_model_name.lower():
                logger.info("Using INT8 ONNX reranker for %s", self.reranker_model_name)
                self._reranker_model = ONNXReranker()
            else:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                logger.info("Loading FastEmbed reranking model %s", self.reranker_model_name)
                self._reranker_model = TextCrossEncoder(model_name=self.reranker_model_name)
        return self._reranker_model

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of numpy arrays representing the embeddings.
        """
        if not texts:
            return []
        return list(self.embedder.embed(texts))

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string.

        Args:
            query: Query string to embed.

        Returns:
            Numpy array representing the embedding.
        """
        res = self.embedder.embed([query])
        if hasattr(res, "__next__") or hasattr(res, "__iter__"):
            return next(iter(res))
        return res[0]

    def rerank(
        self, query: str, documents: list[str], top_k: int = 5
    ) -> list[RerankResult]:
        """Rerank documents against a query using the cross-encoder.

        Args:
            query: The search query.
            documents: List of document strings to rerank.
            top_k: Number of top results to return.

        Returns:
            List of RerankResult sorted by score descending.
        """
        if not documents:
            return []

        raw_output = list(self.reranker.rerank(query, documents))

        # Handle raw float scores or objects with .score/.corpus_id
        results: list[RerankResult] = []
        for i, item in enumerate(raw_output):
            if isinstance(item, (float, int, np.floating)):
                score = float(item)
                idx = i
            elif hasattr(item, "score") and hasattr(item, "corpus_id"):
                score = float(item.score)
                idx = item.corpus_id
            elif hasattr(item, "score"):
                score = float(item.score)
                idx = getattr(item, "index", i)
            else:
                score = float(item)
                idx = i

            results.append(
                RerankResult(index=idx, score=score, text=documents[idx])
            )

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
