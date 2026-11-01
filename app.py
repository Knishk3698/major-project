"""Simple Streamlit interface for the project RAG demonstration."""

import os
import re
from pathlib import Path

import requests
import streamlit as st

from src.embedding import EmbeddingModel
from src.pdf_processor import chunk_text, extract_text_from_pdf
from src.retrieval import (
    create_faiss_index,
    load_vectorstore,
    retrieve_chunks,
    save_vectorstore,
)


PROJECT_ROOT = Path(__file__).resolve().parent
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"


def load_env_file():
    """Read simple KEY=value settings from .env without another dependency."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(name.strip(), value)


def generate_grounded_answer(question, retrieved_chunks):
    """Call a user-configured OpenAI-compatible chat-completions endpoint."""
    load_env_file()
    api_url = os.getenv("LLM_API_URL", "").strip()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model_name = os.getenv("LLM_MODEL", "").strip()
    if not api_url or not api_key or not model_name:
        raise RuntimeError(
            "Configure LLM_API_URL, LLM_API_KEY, and LLM_MODEL in a .env file. "
            "Use an API endpoint that supports the OpenAI-compatible chat completions format."
        )

    context = "\n\n".join(
        f"[Source: {chunk.get('source', 'unknown')}, page {chunk.get('page_number', '?')}]\n"
        f"{chunk['text']}"
        for chunk in retrieved_chunks
    )
    response = requests.post(
        api_url,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model_name,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer using only the supplied document context. "
                        "If the context does not contain the answer, say that it is "
                        "not available in the document. Do not use outside knowledge."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Document context:\n{context}\n\nQuestion: {question}",
                },
            ],
        },
        timeout=90,
    )
    response.raise_for_status()
    answer = response.json()["choices"][0]["message"]["content"]
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("The configured LLM returned an empty answer.")
    return answer.strip()


def check_faithfulness(answer, context):
    """Estimate lexical support; this is not a full hallucination detector."""
    ignored_words = {
        "about", "after", "again", "against", "also", "among", "because",
        "before", "being", "between", "could", "does", "during", "from",
        "have", "into", "more", "most", "only", "other", "over", "same",
        "should", "some", "such", "than", "that", "their", "them", "then",
        "there", "these", "they", "this", "those", "through", "under",
        "very", "were", "what", "when", "where", "which", "while", "with",
        "would", "your",
    }
    answer_words = {
        word.lower()
        for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", answer)
        if len(word) > 2 and word.lower() not in ignored_words
    }
    context_words = {
        word.lower()
        for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", context)
    }
    if not answer_words:
        return 0.0
    return len(answer_words & context_words) / len(answer_words)


@st.cache_resource
def get_embedding_model():
    return EmbeddingModel()


def main():
    st.set_page_config(page_title="Document RAG Assistant", page_icon="📄")
    st.title("RAG-Based Domain Expert Assistant")
    st.write(
        "Upload a PDF, retrieve relevant passages, and ask questions grounded in that document."
    )

    uploaded_pdf = st.file_uploader("Choose a PDF document", type=["pdf"])
    if st.button("Process document", type="primary", disabled=uploaded_pdf is None):
        try:
            with st.spinner("Extracting, chunking, embedding, and indexing the PDF..."):
                pages = extract_text_from_pdf(
                    uploaded_pdf.getvalue(), source_name=uploaded_pdf.name
                )
                chunks = chunk_text(pages, chunk_size=500, overlap=100)
                if not chunks:
                    raise ValueError("No text chunks were produced from this PDF.")
                embedding_model = get_embedding_model()
                embeddings = embedding_model.generate_embeddings(
                    [chunk["text"] for chunk in chunks]
                )
                index = create_faiss_index(embeddings)
                save_vectorstore(index, chunks, VECTORSTORE_DIR)
                st.session_state["index"] = index
                st.session_state["chunks"] = chunks
            st.success(f"Processed {len(pages)} pages into {len(chunks)} chunks.")
        except Exception as error:
            st.error(f"Could not process the PDF: {error}")

    index = st.session_state.get("index")
    chunks = st.session_state.get("chunks")
    if index is None:
        try:
            index, chunks = load_vectorstore(VECTORSTORE_DIR)
            st.session_state["index"] = index
            st.session_state["chunks"] = chunks
            st.caption(f"Loaded {len(chunks)} saved chunks from the local vectorstore.")
        except FileNotFoundError:
            st.info("Process a PDF to create a local search index.")

    question = st.text_input("Ask a question about the processed document")
    if st.button("Ask", disabled=not question.strip()):
        index = st.session_state.get("index")
        chunks = st.session_state.get("chunks")
        if index is None:
            st.warning("Process a PDF before asking a question.")
            return

        try:
            with st.spinner("Finding relevant passages..."):
                retrieved = retrieve_chunks(
                    question, index, chunks, get_embedding_model(), top_k=3
                )
            context = "\n\n".join(chunk["text"] for chunk in retrieved)
            st.subheader("Answer")
            try:
                answer = generate_grounded_answer(question, retrieved)
                st.write(answer)
                score = check_faithfulness(answer, context)
                st.caption(
                    f"Rough word-overlap support estimate: {score:.0%}. "
                    "This simple check is only a clue and does not detect every hallucination."
                )
            except Exception as error:
                st.error(f"Answer generation is unavailable: {error}")

            st.subheader("Retrieved passages")
            for number, chunk in enumerate(retrieved, start=1):
                with st.expander(
                    f"Passage {number}: {chunk.get('source', 'unknown')}, "
                    f"page {chunk.get('page_number', '?')} "
                    f"(similarity {chunk['score']:.3f})"
                ):
                    st.write(chunk["text"])
        except Exception as error:
            st.error(f"Could not retrieve document passages: {error}")


if __name__ == "__main__":
    main()
