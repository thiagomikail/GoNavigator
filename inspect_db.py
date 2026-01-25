import chromadb
# Backward compatibility logic like in main.py
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
except AttributeError:
    from chromadb.config import Settings
    chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db"))

try:
    collection = chroma_client.get_collection(name="trip_knowledge")
    # Get all metadata
    data = collection.get()
    
    if not data['ids']:
        print("Database is EMPTY.")
    else:
        metadatas = data['metadatas']
        trip_ids = set(m.get('trip_id', 'UNKNOWN') for m in metadatas)
        print(f"Found {len(data['ids'])} chunks.")
        print("Available Trip IDs:", trip_ids)

except Exception as e:
    print(f"Error reading DB: {e}")
