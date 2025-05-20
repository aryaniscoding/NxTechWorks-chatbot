from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('BAAI/bge-small-en-v1.5')

def rerank_chunks(query: str, candidates: list[str], top_k=5):
    q_emb = model.encode(query, convert_to_tensor=True)
    c_emb = model.encode(candidates, convert_to_tensor=True)
    sims = util.cos_sim(q_emb, c_emb)[0]
    vals, idxs = sims.topk(top_k)
    return [candidates[i] for i in idxs.cpu().numpy()]
