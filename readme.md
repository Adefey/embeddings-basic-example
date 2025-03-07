## Basic embeddings usage

Wrapper for feature exctraction model with all enums for possible parameters

# Used model
jinaai/jina-embeddings-v3

https://huggingface.co/jinaai/jina-embeddings-v3

# Example

Create model, make 2 empeddings, calculate similarity

```
from embeddings import Embeddings

model = Embeddings()
e1 = model.encode_text("Hello")
e2 = model.encode_text("Здравствуй")
cos_sim = model.cosine_similarity(e1, e2)
cos_sim_norm = model.normalized_cosine_similarity(e1, e2)

print(cos_sim)
print(type(cos_sim_norm))
```