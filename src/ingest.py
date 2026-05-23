from sentence_transformers import SentenceTransformer
import pdfplumber, uuid, os, json, numpy as np
MODEL = SentenceTransformer("all-MiniLM-L6-v2")
INDEX_PATH = "index.npy"
META_PATH = "index_meta.json"
DOC_STORE = "docs/"

def ingest_document(uploaded_file):
    os.makedirs(DOC_STORE, exist_ok=True)
    doc_id = str(uuid.uuid4())
    path = os.path.join(DOC_STORE, f"{doc_id}.txt")
    if uploaded_file.type == "application/pdf" or uploaded_file.name.endswith(".pdf"):
        with pdfplumber.open(uploaded_file) as pdf, open(path,"w",encoding="utf-8") as out:
            for p in pdf.pages:
                text = p.extract_text() or ""
                out.write(text + "\n\n")
    else:
        text = uploaded_file.getvalue().decode("utf-8")
        open(path,"w",encoding="utf-8").write(text)
    texts = [t for t in open(path,"r",encoding="utf-8").read().split("\n\n") if t.strip()]
    embeddings = MODEL.encode(texts)
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        existing = np.load(INDEX_PATH)
        meta = json.load(open(META_PATH,encoding="utf-8"))
        all_embeddings = np.vstack([existing, embeddings])
        meta.extend([{"doc_id": doc_id, "chunk": t} for t in texts])
    else:
        all_embeddings = embeddings
        meta = [{"doc_id": doc_id, "chunk": t} for t in texts]
    np.save(INDEX_PATH, all_embeddings)
    with open(META_PATH,"w",encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    meta_doc = {"doc_id": doc_id, "path": path, "chunks": texts}
    with open(os.path.join(DOC_STORE,f"{doc_id}.json"),"w",encoding="utf-8") as f:
        json.dump(meta_doc,f, ensure_ascii=False)
    return doc_id
