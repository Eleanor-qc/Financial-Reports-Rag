# app.py
import os
import streamlit as st

INDEX_PATH = "data/financial_reports_faiss.pkl"

st.set_page_config(page_title="Financial RAG", layout="wide")
st.title("Financial Reports RAG")

@st.cache_resource
def ensure_index_built():
    """
    This function runs only once per container.
    If the FAISS index does not exist, build it.
    """
    if not os.path.exists(INDEX_PATH):
        import build_faiss_index
        build_faiss_index.main()

    # Import AFTER index exists
    from rag_core import answer_user_question
    return answer_user_question

answer_user_question = ensure_index_built()

query = st.text_input(
    "Ask a question about the financial reports:",
    placeholder="e.g. What factors drove AWS revenue growth?"
)

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer = answer_user_question(query)
            st.subheader("Answer")
            st.write(answer)