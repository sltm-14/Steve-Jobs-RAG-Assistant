# To Run
# python -m app.ingest_corpus

from app.rag import chunk_text_by_paragraphs, clean_text
from sentence_transformers import SentenceTransformer
from pathlib import Path

import chromadb


# Load embedding model for transforming text into vectors
model = SentenceTransformer("all-MiniLM-L6-v2")

def ingest_file(content, file_name):

    processed_text = clean_text(content)

    new_path = f"data/processed/{file_name}"

    try:
        with open(new_path, "w", encoding="utf-8")  as archivo:
            archivo.write(processed_text)
            print("New file created")
        
    except FileExistsError:
        raise FileExistsError("The file already exists")
    
    return chunking_process(processed_text, file_name)

def ingest_from_path(file_path: str, source: str):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            print("File read")

    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"No se encontró el archivo: {file_path}"
        ) from e
    
    processed_text = clean_text(text)

    file_name = Path(file_path).name

    new_path = f"data/processed/{file_name}"
    try:
        with open(new_path, "w", encoding="utf-8")  as archivo:
            archivo.write(processed_text)
            print("New file created")
        
    except FileExistsError:
        raise FileExistsError("The file already exists")

    return chunking_process(processed_text, file_name)

def chunking_process(processed_text, source):
    chunks = chunk_text_by_paragraphs(processed_text,1000)

    # Create or connect to persistent ChromaDB storage
    client = chromadb.PersistentClient(path="./chroma_db")

    # Create or get the collection where documents will be stored
    collection = client.get_or_create_collection(name="steve_jobs_corpus")

    embeddings = model.encode(chunks).tolist() # Convert chunks into embeddings
    current_count = collection.count() # Count existing documents to generate unique IDs

    # Generate IDs for each chunk
    ids = [f"jobs_chunk_{current_count + i}" for i in range(len(chunks))]

    # Add metadata for each chunk
    metadatas = [{"source": source} for _ in chunks]

    # Store documents, embeddings, metadata, and IDs in ChromaDB
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    # Return ingestion result
    return {
        "message": "Chunks ingested successfully",
        "num_chunks": len(chunks),
        "ids count": len(ids)
    }
    
