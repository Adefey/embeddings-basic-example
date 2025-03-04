import numpy as np
from enum import IntEnum, StrEnum
from transformers import AutoModel
import time


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
    ):
        self.checkpoint = checkpoint
        self.task = task
        self.max_length = max_length
        self.embedding_dims = embedding_dims
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


def similarity(e1: np.ndarray, e2: np.ndarray):
    return np.dot(e1, e2)


emb = Embeddings()
start = time.time_ns()
e1 = emb.encode_text("Всем привет, меня зовут Жопа")
end = time.time_ns()
print((end - start) / 1_000_000_000)

start = time.time_ns()
e2 = emb.encode_text("Здравствуйте, меня зовут Андрей")
end = time.time_ns()
print((end - start) / 1_000_000_000)

start = time.time_ns()
e3 = emb.encode_text("Всем привет, меня зовут не Жёпа")
end = time.time_ns()
print((end - start) / 1_000_000_000)

start = time.time_ns()
e4 = emb.encode_text(
    "На текущий момент цены на бензин продолжают демонстрировать незначительный"
    " рост, несмотря на некоторые колебания на рынке нефти.Так, по состоянию на"
    " 20 февраля 2025 года, стоимость литра бензина марки АИ-92 составляла в"
    " среднем 56,2 рубля, в то время как АИ-95 оценивался в 62,5 рубля за литр."
    " За последнюю неделю наблюдался незначительное повышение цен на бензин в"
    " пределе 2−3%, замечает эксперт.«В марте цены продолжат свой восходящий"
    " тренд, увеличившись на дополнительные 5−7 процентов от текущих значений."
    " Этот рост, хотя и существенный, не является критически значимым, так как"
    " связан он с ожидаемым увеличением спроса в преддверии весенне-летнего"
    " сезона», — указал Вавилов. Стоит отметить, что цены на нефть, играющие"
    " ключевую роль в формировании стоимости топлива, остаются на относительно"
    " стабильном уровне, что способствует сдерживанию значительных колебаний"
    " цен на розничном уровне.«Однако глобальные геополитические события и"
    " изменения в мировой экономике могут оказать дополнительное давление на"
    " цены в среднесрочной перспективе», — разъяснил Николай Вавилов."
)
end = time.time_ns()
print((end - start) / 1_000_000_000)

start = time.time_ns()
e5 = emb.encode_text("Бензин подорожал")
end = time.time_ns()
print((end - start) / 1_000_000_000)

start = time.time_ns()
e6 = emb.encode_text("Бензин подешевел")
end = time.time_ns()
print((end - start) / 1_000_000_000)


print("e1 e2 sim", similarity(e1, e2))
print("e1 e3 sim", similarity(e1, e3))
print("e1 e4 sim", similarity(e1, e4))
print("e4 e5 sim", similarity(e4, e5))
print("e4 e6 sim", similarity(e4, e6))
