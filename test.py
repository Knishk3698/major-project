
from pathlib import Path

from src.pdf_processor import extract_text_from_pdf, chunk_text

PROJECT_ROOT = Path(__file__).resolve().parent
PDF_PATH = PROJECT_ROOT / "data" / "documents" / "nist_incident_response.pdf"

# Step 1: Extract text
pages = extract_text_from_pdf(PDF_PATH)

print("PDF extraction successful!")
print("Total pages containing text:", len(pages))

# Step 2: Display first page text
print("\nFirst page preview:")
print(pages[0]["text"][:500])

# Step 3: Add source information
for page in pages:
    page["source"] = str(PDF_PATH)

# Step 4: Create chunks
chunks = chunk_text(pages, chunk_size=500, overlap=100)

print("\nChunking successful!")
print("Total chunks generated:", len(chunks))

# Step 5: Display first chunk
print("\nFirst chunk:")
print(chunks[0]["text"][:500])

print("\nSource:", chunks[0]["source"])
print("Page number:", chunks[0]["page_number"])
