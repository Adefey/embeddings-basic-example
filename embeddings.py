import numpy as np
from enum import IntEnum, StrEnum
from transformers import AutoModel
import torch


class Embeddings:
    """
    Wrapper for XLMRobertaLoRA
    """

    class EmbeddingDim(IntEnum):
        """
        Size of embedding
        """

        DIM32 = 32
        DIM64 = 64
        DIM128 = 128
        DIM256 = 256
        DIM512 = 512
        DIM768 = 768
        DIM1024 = 1024

    class Task(StrEnum):
        """
        Task passed to encode methods
        """

        RETRIEVAL_QUERY = "retrieval.query"
        RETRIEVAL_PASSAGE = "retrieval.passage"
        SEPARATION = "separation"
        CLASSIFICATION = "classification"
        TEXT_MATCHING = "text-matching"

    class Device(StrEnum):
        """
        Device to run model on
        """

        CPU = "cpu"
        CUDA = "cuda"
        MPS = "mps"
        XPU = "xpu"
        XLA = "xla"
        META = "meta"

    def __init__(
        self,
        checkpoint: str = "jinaai/jina-embeddings-v3",
        task: Task = Task.TEXT_MATCHING,
        max_length: int = 8192,
        seed: int = 14565,
        embedding_dims: EmbeddingDim = EmbeddingDim.DIM1024,
        dtype: torch.dtype = torch.float16,
        device: Device = Device.CPU,
    ):
        self.checkpoint = checkpoint
        self.task = task
        self.max_length = max_length
        self.embedding_dims = embedding_dims
        self.seed = seed
        self.dtype = dtype
        self.device = torch.device(device)
        torch.manual_seed(self.seed)
        self.model = AutoModel.from_pretrained(
            self.checkpoint, trust_remote_code=True, torch_dtype=self.dtype
        ).to(self.device)

    def encode_batch(self, texts: list[str]) -> np.ndarray[np.ndarray]:
        """
        Encode multiple strings, get array of embeddings
        """
        with torch.no_grad():
            result = self.model.encode(
                texts,
                task=self.task,
                max_length=self.max_length,
                truncate_dim=self.embedding_dims,
            )
        return result

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode one string, return one embedding
        """
        with torch.no_grad():
            result = self.model.encode(
                [text],
                task=self.task,
                max_length=self.max_length,
                truncate_dim=self.embedding_dims,
            )[0]
        return result

    def cosine_similarity(self, e1: np.ndarray, e2: np.ndarray) -> float:
        """
        Cosine similarity between two vectors (embeddings), value range: [-1, 1]
        """
        return np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))

    def normalized_cosine_similarity(self, e1: np.ndarray, e2: np.ndarray) -> float:
        """
        Normalized cosine similarity between two vectors (embeddings), value range: [0, 1]
        """
        return (np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))) * 0.5 + 0.5
