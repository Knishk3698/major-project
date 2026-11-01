# RAG-Based Domain Expert Assistant

A small Retrieval-Augmented Generation (RAG) application. It extracts text from a PDF, retrieves relevant passages, and asks a configured OpenAI-compatible language model to answer from those passages.

## Project contents

```text
data/
  documents/nist_incident_response.pdf   NIST SP 800-61r3 sample document
  evaluation/natural_questions.json      500 question-answer evaluation examples
src/
  pdf_processor.py                        PDF extraction and overlapping chunks
  embedding.py                            Sentence Transformer embeddings
  retrieval.py                            FAISS indexing, persistence, and search
vectorstore/
  chunks.faiss                             Prebuilt index for the NIST PDF
  chunks.json                             Matching text and source/page metadata
app.py                                    Streamlit application
download_dataset.py                      Optional Natural Questions sample exporter
test.py                                   PDF extraction and chunking check
test_embedding.py                        PDF-to-embedding check
test_embeddings.py                       PDF-to-embedding check
requirements.txt                          Python dependencies
```

All paths used by the application and scripts are resolved from the repository location. The project does not require a particular local username, machine path, or checked-in virtual environment.

## Windows setup

Install Git and 64-bit Python 3.10, including the Python Launcher (`py`). In PowerShell, run:

```powershell
git clone https://github.com/Knishk3698/major-project.git
cd major-project
py -3.10 --version
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use this command for the current terminal, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The first embedding run downloads `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face. Allow network access to Hugging Face the first time, or pre-cache that model in the environment.

## Configure your LLM

Create a `.env` file in the repository root using your provider's own endpoint, key, and model. The URL must accept OpenAI-compatible chat-completions requests.

```powershell
@'
LLM_API_URL=https://your-provider.example/v1/chat/completions
LLM_API_KEY=replace-with-your-own-key
LLM_MODEL=replace-with-your-model
'@ | Set-Content -Encoding ascii .env
```

Replace all three example values. `.env` is ignored by Git; never commit or share your credential. The app also reads these variables from the process environment. PDF processing and retrieval work without an LLM key, but generating an answer requires a configured endpoint.

## Run the application

From the repository root, with `.venv` activated:

```powershell
python -m streamlit run app.py
```

Open the local URL Streamlit prints. The checked-in vectorstore contains the NIST sample's chunks and is loaded when the app starts. You can also upload another PDF and select **Process document**; this replaces the local `vectorstore/chunks.faiss` and `vectorstore/chunks.json` with that PDF's index. Ask a question to see the retrieved passages and, when configured, an answer grounded in them.

## Run the checks

Run these from the repository root:

```powershell
python test.py
python test_embedding.py
python test_embeddings.py
```

`test.py` checks PDF extraction and chunk creation. The two embedding scripts encode the NIST PDF's chunks and verify that each vector has 384 dimensions. The embedding checks need the model download on first use. They do not call an LLM API. Answer generation in the app requires your configured external LLM service and API key.

## Data and RAG pipeline

The PDF pipeline is: PyMuPDF extracts text with page numbers; `chunk_text` makes overlapping chunks of up to 500 words; Sentence Transformers encodes each chunk into 384 values; FAISS stores L2-normalized vectors and searches them by inner product (cosine similarity). The matching chunk text, source filename, and page number are stored in `vectorstore/chunks.json`. The app embeds a question, retrieves up to three passages, and sends those passages and the question to the configured LLM.

`data/evaluation/natural_questions.json` is a separate sample of 500 Natural Questions question-answer pairs. The app does **not** currently load or use this dataset for indexing, retrieval, prompting, or scoring. It is included for manual evaluation or future evaluation tooling. `download_dataset.py` can export a fresh 500-record sample from the `sentence-transformers/natural-questions` dataset on Hugging Face; it requires internet access and the `datasets` dependency. Running it replaces the tracked JSON sample, so review the result before committing.

The displayed word-overlap score is a rough lexical estimate of answer support, not a reliable hallucination detector. Review the retrieved passages alongside each answer.

## Credentials and generated files

`.venv/`, `.env`, Python bytecode, and `__pycache__/` are excluded by `.gitignore`. The PDF, evaluation JSON, source files, requirements, README, and prebuilt vectorstore are intentionally tracked. Do not add personal credentials or machine-specific paths to source files.
