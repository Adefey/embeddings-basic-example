from embeddings import (
    Embeddings,
    cosine_similarity,
    normalized_cosine_similarity,
)

model = Embeddings()
e1 = model.encode_text("Hello")
e2 = model.encode_text("Здравствуй")
cos_sim = cosine_similarity(e1, e2)
cos_sim_norm = normalized_cosine_similarity(e1, e2)

print(cos_sim)
print(cos_sim_norm)
