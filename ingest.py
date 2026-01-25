import os
import argparse
import chromadb
import google.generativeai as genai
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in .env file")

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Vector DB
# robust_client handles concurrency better for production-lite
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="trip_knowledge")

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def ingest_pdf(pdf_path: str, trip_id: str):
    print(f"Processing {pdf_path} for Trip ID: {trip_id}...")
    
    # 1. Extract
    full_text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(full_text)} characters.")
    
    # 2. Chunk
    chunks = chunk_text(full_text)
    print(f"Created {len(chunks)} chunks.")
    
    # 3. Embed & Store
    # We use the 'trip_id' in metadata to filter later.
    ids = [f"{trip_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": pdf_path, "trip_id": trip_id} for _ in chunks]
    
    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )
    print(f"Ingestion Complete for Trip {trip_id}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a PDF for a specific Trip Code.")
    parser.add_argument("--file", required=True, help="Path to the PDF file")
    parser.add_argument("--trip-id", required=True, help="Unique Trip Code (e.g., PARIS24)")
    
    args = parser.parse_args()
    
    if os.path.exists(args.file):
        ingest_pdf(args.file, args.trip_id)
    else:
        print(f"Error: File '{args.file}' not found.")
