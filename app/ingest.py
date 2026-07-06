# Ingestion logic for the RAG system.
# This module handles text cleaning, chunking, embedding generation,
# and storing chunks in ChromaDB.

import logging
import uuid

from pathlib import Path
from io import BytesIO
from pypdf import PdfReader


logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Basic text cleaning before chunking.
    Removes:
    - empty lines
    - very short noisy lines
    - lines containing known editorial/source artifacts
    """

    lines = text.splitlines()
    forbidden_words = [
        "Copyright",
        "Typeset",
        "Credits",
        "Published"
    ]

    cleaned_lines = []

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Skip very short lines that are likely noise
        if len(line) <= 14:
            continue

        # Skip lines containing unwanted metadata/editorial artifacts
        if any(word in line for word in forbidden_words):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

def chunk_text_by_paragraphs(text: str, max_chars: int = 1000):
    """
    Split text into chunks while trying to preserve paragraph boundaries.
    This avoids cutting words or sentences in the middle as often as
    character-based chunking.
    """

    chunks = []
    lines = text.splitlines(keepends=True)
    current_chunk = ""

    for line in lines:
        # If a single line is longer than max_chars, store it as its own chunk.
        # This is a simple fallback for very long paragraphs.
        if len(line) > max_chars:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            chunks.append(line)
            continue

        # Add line to current chunk if it still fits
        if len(current_chunk) + len(line) <= max_chars:
            current_chunk += line

        # Otherwise, save current chunk and start a new one
        else:
            if current_chunk:
                chunks.append(current_chunk)

            current_chunk = line

    # Save final remaining chunk

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from a PDF file.

    This works for PDFs with selectable text.
    It does not perform OCR on scanned images.
    """

    reader = PdfReader(BytesIO(file_bytes)) # Create a PDF reader object from the file bytes

    pages_text = [] # List to store the text of each page

    for page_number, page in enumerate(reader.pages, start=1):
        # Extract the text from the page, or an empty string if there is no text
        text = page.extract_text() or "" 

        # If the text is not empty, add it to the list
        if text.strip(): 
            pages_text.append(f"\n--- Page {page_number} ---\n{text}") # Add the text to the list, with a header for the page number

    return "\n".join(pages_text) # Join the text of all pages with a newline


def ingest_file(content: str, file_name: str, model, client, collection):
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
    file_stem = Path(file_name).stem
    new_path = f"data/processed/{file_stem}.txt"

    with open(new_path, "w", encoding="utf-8") as archivo:
        archivo.write(processed_text)
        logger.info("Processed file saved. path=%s", new_path)

    # Chunk, embed, and store content in ChromaDB
    return chunking_process(processed_text, file_name, model, client, collection)


def chunking_process(processed_text: str, source: str, model, client, collection):
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
    logger.info("Text chunked. source=%s chunks=%d", source, len(chunks))

    # Generate embeddings for all chunks
    embeddings = model.encode(chunks).tolist()
    logger.info("Embeddings generated. count=%d", len(embeddings))

    # Generate IDs for each new chunk
    ids = [str(uuid.uuid4()) for _ in chunks]

    # Attach source metadata to every chunk
    metadatas = [{"source": source} for _ in chunks]

    # Store chunks, embeddings, metadata, and IDs in ChromaDB
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    logger.info("Chunks stored in ChromaDB. source=%s total_ids=%d", source, len(ids))

    return {
        "message": "Chunks ingested successfully",
        "num_chunks": len(chunks),
        "ids_count": len(ids)
    }


def ingest_from_path(file_path: str, source: str, model, client, collection):
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
            logger.info("File read. path=%s chars=%d", file_path, len(text))

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
        logger.info("Processed file saved. path=%s", new_path)

    # Chunk, embed, and store content in ChromaDB
    return chunking_process(processed_text, source, model, client, collection)


