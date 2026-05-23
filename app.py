import streamlit as st
from src import ingest, rag, ecm_export, utils
import os

st.set_page_config(page_title="Regulatory RAG Demo", layout="wide")
st.title("Generics Regulatory RAG Prototype — eCTD Scaffold & QA")

st.markdown("Demo: upload a small SOP or dossier, ingest to build embeddings, ask RAG Q&A, and generate a sample eCTD scaffold.")

with st.sidebar:
	st.header("Upload / Ingest")
	uploaded = st.file_uploader("Upload dossier PDF or SOP", type=["pdf", "txt", "docx"])
	if st.button("Ingest"):
		if uploaded:
			try:
				doc_id = ingest.ingest_document(uploaded)
				st.success(f"Ingested: {doc_id}")
			except Exception as e:
				st.error(f"Ingest failed: {e}")
		else:
			st.error("Upload a file first")

st.header("RAG Q&A")
query = st.text_area("Ask the QA assistant about regulatory requirements or draft text", height=140)
if st.button("Get Answer"):
	if not query.strip():
		st.warning("Enter a question.")
	else:
		with st.spinner("Retrieving..."):
			try:
				ans = rag.answer_query(query)
			except Exception as e:
				ans = {"answer": f"Error calling RAG: {e}", "sources": []}
		st.subheader("Answer")
		st.write(ans.get("answer", "No answer"))
		st.subheader("Citations / Sources")
		for s in ans.get("sources", []):
			st.markdown(f"- {s.get('source','unknown')} — {s.get('snippet','')[:400]}")

st.header("eCTD Scaffold")
if st.button("Generate eCTD Scaffold (sample)"):
	try:
		scaffold = ecm_export.generate_scaffold()
		st.download_button("Download scaffold (JSON)", data=utils.to_json(scaffold), file_name="scaffold.json")
	except Exception as e:
		st.error(f"Scaffold generation failed: {e}")

st.sidebar.markdown("### Workspace status")
docs_dir = "docs"
exists = os.path.exists(docs_dir) and any(os.scandir(docs_dir))
st.sidebar.write("Docs folder present:", os.path.exists(docs_dir))
st.sidebar.write("Docs contain files:", bool(exists))
st.sidebar.write("Index file present:", os.path.exists("index.npy"))

st.markdown("Usage note: For full LLM responses set OPENAI_API_KEY in Streamlit Secrets. Do not upload confidential files to this demo.")



