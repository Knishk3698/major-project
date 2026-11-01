
from src.pdf_processor import extract_text_from_pdf, chunk_text
from src.embedding import EmbeddingModel

PDF_PATH = "data/documents/nist_incident_response.pdf"

# 1. Extract PDF text
pages = extract_text_from_pdf(PDF_PATH)

# Add source filename
for page in pages:
    page["source"] = PDF_PATH

# 2. Generate text chunks
chunks = chunk_text(pages, chunk_size=500, overlap=100)

print(f"\nTotal chunks: {len(chunks)}")

# 3. Extract the text from each chunk
texts = [chunk["text"] for chunk in chunks]

# 4. Load the embedding model
embedding_model = EmbeddingModel()

# 5. Generate embeddings
embeddings = embedding_model.generate_embeddings(texts)

# 6. Display results
print("\nEmbedding generation successful!")

print("Number of embeddings:", embeddings.shape[0])
print("Dimensions per embedding:", embeddings.shape[1])

print("\nFirst embedding:")
print(embeddings[0])


