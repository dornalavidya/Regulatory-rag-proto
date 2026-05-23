import os, json, pickle
import numpy as np, openai
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances
INDEX_PATH = "index.npy"
META_PATH = "index_meta.json"
VECT_PATH = "vectorizer.pkl"
openai.api_key = os.getenv("OPENAI_API_KEY","")

def _retrieve(query, k=4):
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH) or not os.path.exists(VECT_PATH):
        return []
    embeddings = np.load(INDEX_PATH)
    meta = json.load(open(META_PATH,encoding="utf-8"))
    with open(VECT_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    q_emb = vectorizer.transform([query]).toarray()
    nn = NearestNeighbors(n_neighbors=min(k, len(embeddings)), metric='cosine').fit(embeddings)
    distances, indices = nn.kneighbors(q_emb)
    snippets = []
    for idx in indices[0]:
        item = meta[int(idx)]
        snippets.append({"source": item.get("doc_id"), "snippet": item.get("chunk")[:1000]})
    return snippets

def answer_query(query):
    sources = _retrieve(query)
    context = "\n\n".join([s["snippet"] for s in sources])
    prompt = f"You are a Regulatory QA Assistant. Use the context below to answer the question. Cite sources by doc id.\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer with bullet points and include source citations."
    if not openai.api_key:
        return {"answer":"OPENAI_API_KEY not set in environment. Set it in Streamlit secrets.", "sources": sources}
    resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], max_tokens=800)
    return {"answer": resp["choices"][0]["message"]["content"], "sources": sources}
