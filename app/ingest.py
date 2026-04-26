# Ingestion logic for the RAG system.
# This module handles text cleaning, chunking, embedding generation,
# and storing chunks in ChromaDB.

from app.rag import chunk_text_by_paragraphs, clean_text
from sentence_transformers import SentenceTransformer
from pathlib import Path

import chromadb


# Load embedding model once when this module is imported.
# This model converts text chunks into embedding vectors.
model = SentenceTransformer("all-MiniLM-L6-v2")


def ingest_file(content: str, file_name: str):
    """
    Ingest text content received from an uploaded file.

    Args:
        content: Raw text extracted from the uploaded file.
        file_name: Name of the uploaded file.

    Returns:
        Summary of the ingestion process.
    """

    # Clean raw text before chunking
    processed_text = clean_text(content)

    # Save cleaned version for debugging / inspection
    new_path = f"data/processed/{file_name}"

    with open(new_path, "w", encoding="utf-8") as archivo:
        archivo.write(processed_text)
        print("Processed file created")

    # Chunk, embed, and store content in ChromaDB
    return chunking_process(processed_text, file_name)


def ingest_from_path(file_path: str, source: str):
    """
    Ingest a local text file from disk.

    This is useful for batch ingestion of the initial corpus without using
    the FastAPI upload endpoint.

    Args:
        file_path: Path to the local raw text file.
        source: Source name to store as metadata.

    Returns:
        Summary of the ingestion process.
    """

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            print("File read")

    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"No se encontró el archivo: {file_path}"
        ) from e

    # Clean text before chunking
    processed_text = clean_text(text)

    # Use original file name for processed output
    file_name = Path(file_path).name
    new_path = f"data/processed/{file_name}"

    # Save cleaned text for inspection
    with open(new_path, "w", encoding="utf-8") as archivo:
        archivo.write(processed_text)
        print("Processed file created")

    # Chunk, embed, and store content in ChromaDB
    return chunking_process(processed_text, source)


def chunking_process(processed_text: str, source: str):
    """
    Convert cleaned text into chunks, generate embeddings, and store them.

    Args:
        processed_text: Cleaned text to be indexed.
        source: Source name to attach as metadata to each chunk.

    Returns:
        Dictionary with ingestion summary.
    """

    # Split cleaned text into paragraph-aware chunks
    chunks = chunk_text_by_paragraphs(processed_text, 1000)

    # Connect to persistent ChromaDB storage
    client = chromadb.PersistentClient(path="./chroma_db")

    # Create or retrieve target collection
    collection = client.get_or_create_collection(name="steve_jobs_corpus")

    # Generate embeddings for all chunks
    embeddings = model.encode(chunks).tolist()

    # Count existing chunks to generate unique IDs
    current_count = collection.count()

    # Generate IDs for each new chunk
    ids = [f"jobs_chunk_{current_count + i}" for i in range(len(chunks))]

    # Attach source metadata to every chunk
    metadatas = [{"source": source} for _ in chunks]

    # Store chunks, embeddings, metadata, and IDs in ChromaDB
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    return {
        "message": "Chunks ingested successfully",
        "num_chunks": len(chunks),
        "ids_count": len(ids)
    }