import numpy as np
from enum import IntEnum, StrEnum
from transformers import AutoModel
import time
import torch


class Embeddings:

    class EmbeddingDim(IntEnum):
        DIM32 = 32
        DIM64 = 64
        DIM128 = 128
        DIM256 = 256
        DIM512 = 512
        DIM768 = 768
        DIM1024 = 1024

    class Task(StrEnum):
        RETRIEVAL_QUERY = "retrieval.query"
        RETRIEVAL_PASSAGE = "retrieval.passage"
        SEPARATION = "separation"
        CLASSIFICATION = "classification"
        TEXT_MATCHING = "text-matching"

    def __init__(
        self,
        checkpoint: str = "jinaai/jina-embeddings-v3",
        task: Task = Task.TEXT_MATCHING,
        max_length: int = 8192,
        embedding_dims: EmbeddingDim = EmbeddingDim.DIM1024,
        seed: int = 14565,
    ):
        self.checkpoint = checkpoint
        self.task = task
        self.max_length = max_length
        self.embedding_dims = embedding_dims
        self.seed = seed
        torch.manual_seed(self.seed)
        self.model = AutoModel.from_pretrained(self.checkpoint, trust_remote_code=True)

    def encode_batch(self, texts: list[str]):
        result = self.model.encode(
            texts,
            task=self.task,
            max_length=self.max_length,
            truncate_dim=self.embedding_dims,
        )
        return result

    def encode_text(self, text: str):
        result = self.model.encode(
            [text],
            task=self.task,
            max_length=self.max_length,
            truncate_dim=self.embedding_dims,
        )[0]
        return result

    def similarity(self, e1: np.ndarray, e2: np.ndarray):
        return np.dot(e1, e2)
