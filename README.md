# RAG-Based Domain Expert Assistant

A small Retrieval-Augmented Generation (RAG) demonstration. The app reads a PDF, finds the most relevant passages for a question, and asks a configured language model to answer from those passages. It also displays the passages and a basic word-overlap support estimate.

## Project layout

```text
major/
├── data/
│   ├── documents/       # Source PDFs, including the NIST sample
│   └── evaluation/      # Natural Questions QA sample
├── vectorstore/         # Saved FAISS index and chunk metadata
├── src/
│   ├── pdf_processor.py # PDF text extraction and 500-word chunking
│   ├── embedding.py     # 384-dimensional Sentence Transformer vectors
│   └── retrieval.py    # FAISS build, save, load, and search functions
├── app.py              # Streamlit user interface and answer generation
├── test.py             # PDF extraction/chunking demonstration
├── test_embeddings.py  # PDF-to-embedding smoke test
└── download_dataset.py # Downloads the small QA evaluation sample
```

## RAG information flow

1. The user uploads a PDF in Streamlit.
2. PyMuPDF extracts text page by page. The processor creates chunks of up to about 500 words with 100 words of overlap and keeps the source filename and page number.
3. Sentence Transformers (`all-MiniLM-L6-v2`) turns each chunk into a 384-number vector.
4. FAISS stores normalized vectors for cosine-similarity search. The index and chunk metadata are saved in `vectorstore/`.
5. The user asks a question. The same embedding model vectorizes it, and FAISS returns the three closest chunks.
6. If an API endpoint is configured, the app sends the question and those passages to the LLM with instructions to use only that context. Retrieved passages and their page details remain visible in the app.
7. The app shows a simple word-overlap estimate between the answer and retrieved context. It is only a rough clue; it is not a reliable or complete hallucination detector.

## Technologies

- Python
- PyMuPDF for PDF text extraction
- Sentence Transformers with `sentence-transformers/all-MiniLM-L6-v2`
- FAISS CPU for local similarity search
- Streamlit for the web interface
- Hugging Face Datasets for the existing Natural Questions sample downloader
- Requests for calls to an explicitly configured OpenAI-compatible chat-completions API

## Setup

The project uses its existing Windows virtual environment. From the project root, install the listed dependencies if needed:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The first embedding run downloads `all-MiniLM-L6-v2` from Hugging Face. Keep the NIST sample at `data/documents/nist_incident_response.pdf`; the Natural Questions sample is stored at `data/evaluation/natural_questions.json`.

## Configure answer generation

The project does not assume that you have an account or API key with a particular LLM provider. Choose an endpoint and model that you have access to and that accepts the OpenAI-compatible chat-completions request format, then create a local `.env` file in the project root containing:

```text
LLM_API_URL=https://your-provider.example/v1/chat/completions
LLM_API_KEY=your-own-key
LLM_MODEL=your-model-name
```

Replace all three example values with the endpoint, credential, and model supplied by your provider. Do not commit `.env` or share its key. No credentials are included in this project. Without these settings, PDF processing and retrieval still work, but answer generation reports that configuration is needed.

## Run the application

From the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Upload a PDF and select **Process document**. After processing, enter a question and select **Ask**. The app returns an answer when configured and expands the top three retrieved passages with their source pages.

The local vectorstore contains `chunks.faiss` and `chunks.json`. Processing another PDF replaces those saved files with the new document's index.

## Run component checks

PDF extraction and chunking:

```powershell
.\.venv\Scripts\python.exe test.py
```

Embedding dimensions:

```powershell
.\.venv\Scripts\python.exe .\test_embeddings.py
```

The embedding check should report 384 dimensions. FAISS retrieval can also be exercised by processing a PDF in the app and asking a question. Answer generation requires your own configured API endpoint and credentials.

## File guide

- `src/pdf_processor.py`: extracts page text and creates overlapping word chunks.
- `src/embedding.py`: loads the pretrained embedding model and encodes text.
- `src/retrieval.py`: builds cosine-similarity FAISS indexes and stores matching chunk metadata.
- `app.py`: connects upload, processing, indexing, retrieval, answer generation, and the rough support estimate.
- `test.py`: prints an extraction and chunking example from the NIST PDF.
- `test_embeddings.py`: checks that each PDF chunk receives one 384-dimensional vector.
- `download_dataset.py`: downloads 500 Natural Questions question-answer records for evaluation.
