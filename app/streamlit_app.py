import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import streamlit as st

from paperlens.answer import answer
from paperlens.embed import VectorIndex
from paperlens.ingest import pdf_to_chunks
from paperlens.retrieve import retrieve

st.set_page_config(page_title="PaperLens", page_icon="🔎")
st.title("PaperLens")
st.caption("Ask questions about your research papers. Every answer cites the paper and page.")

if "index" not in st.session_state:
    st.session_state.index = VectorIndex()
    st.session_state.papers = set()
    st.session_state.history = []

with st.sidebar:
    st.header("Papers")
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    k = st.slider("Excerpts to use", 2, 8, 4)
    min_score = st.slider("Minimum similarity", 0.0, 0.6, 0.25, 0.01)
    for f in files or []:
        if f.name not in st.session_state.papers:
            with st.spinner(f"Reading {f.name}..."):
                st.session_state.index.add(pdf_to_chunks(f.getvalue(), f.name))
                st.session_state.papers.add(f.name)
    if st.session_state.papers:
        st.write("Loaded:", ", ".join(sorted(st.session_state.papers)))

for role, text, sources in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)
        if sources:
            with st.expander("Sources"):
                for line in sources:
                    st.markdown(line)
        if not st.session_state.papers:
            st.info("Upload one or more research papers in the sidebar, then ask a question. Every answer cites the paper and page.")
question = st.chat_input("Ask a question about the uploaded papers")
if question:
    if not st.session_state.papers:
        st.warning("Upload at least one PDF first.")
    else:
        hits = retrieve(st.session_state.index, question, k=k, min_score=min_score)
        try:
            reply = answer(question, hits)
        except Exception as e:
            reply, hits = f"Error: {e}", []
        sources = [
            f"**[{i}] {c.paper}, page {c.page}** (similarity {s:.2f})\n\n> {c.text}"
            for i, (c, s) in enumerate(hits, start=1)
        ]
        st.session_state.history.append(("user", question, None))
        st.session_state.history.append(("assistant", reply, sources))
        st.rerun()
