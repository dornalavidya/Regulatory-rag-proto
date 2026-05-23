import pdfplumber, uuid, os, json, numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
INDEX_PATH = "index.npy"
META_PATH = "index_meta.json"
VECT_PATH = "vectorizer.pkl"
DOC_STORE = "docs/"

def _read_file_text(path, uploaded_file):
    if uploaded_file.type == "application/pdf" or uploaded_file.name.endswith(".pdf"):
        with pdfplumber.open(uploaded_file) as pdf, open(path,"w",encoding="utf-8") as out:
            for p in pdf.pages:
                text = p.extract_text() or ""
                out.write(text + "\n\n")
    else:
        text = uploaded_file.getvalue().decode("utf-8")
        open(path,"w",encoding="utf-8").write(text)


def ingest_document(uploaded_file):
    os.makedirs(DOC_STORE, exist_ok=True)
    doc_id = str(uuid.uuid4())
    path = os.path.join(DOC_STORE, f"{doc_id}.txt")
    _read_file_text(path, uploaded_file)
    texts = [t for t in open(path,"r",encoding="utf-8").read().split("\n\n") if t.strip()]

    # If there is an existing index, load previous chunks to refit vectorizer
    if os.path.exists(META_PATH):
        meta = json.load(open(META_PATH, encoding="utf-8"))
        existing_chunks = [m.get("chunk") for m in meta]
    else:
        meta = []
        existing_chunks = []

    all_chunks = existing_chunks + texts
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(all_chunks).toarray()

    # Save embeddings and metadata (keep ordering: existing then new)
    np.save(INDEX_PATH, embeddings)
    new_meta = meta + [{"doc_id": doc_id, "chunk": t} for t in texts]
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(new_meta, f, ensure_ascii=False)

    # Save vectorizer
    with open(VECT_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    meta_doc = {"doc_id": doc_id, "path": path, "chunks": texts}
    with open(os.path.join(DOC_STORE, f"{doc_id}.json"), "w", encoding="utf-8") as f:
        json.dump(meta_doc, f, ensure_ascii=False)
    return doc_id
