# Retrieval and response-building logic for the RAG system
from app.llm import generate_answer

def run_retrieval(
    question: str,
    top_k: int,
    mode: str,
    model,
    collection,
    max_distance: float = 0.85
):
    """
    Retrieve relevant chunks from ChromaDB and build an answer.

    Args:
        question: User question.
        top_k: Number of chunks to retrieve from the vector database.
        mode: Response mode. Supported values: "evidence" or "persona".
        model: Embedding model used to encode the question.
        collection: ChromaDB collection.
        max_distance: Maximum allowed distance for retrieved chunks.

    Returns:
        Dictionary with:
        - answer: generated/basic response
        - matches: retrieved chunks that passed the distance filter
    """

    # Validate response mode
    if mode not in ("evidence", "persona"):
        raise ValueError("mode must be 'evidence' or 'persona'")
    
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    
    # Convert user question into an embedding vector
    query_embedding = model.encode([question]).tolist()

    # Search the vector database for the closest chunks
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    # Chroma returns nested lists because it can handle multiple queries at once.
    # Since we only send one question, we take index [0].
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    ids = results["ids"][0]

    # Store only matches that are close enough to the question
    matches = []

    # Iterate through all retrieved results
    for doc, metadata, distance, doc_id in zip(documents, metadatas, distances, ids):
        # Keep only results within the allowed similarity distance threshold
        if distance <= max_distance:
            matches.append({
                "id": doc_id,  # Document identifier
                "text": doc,  # Document content
                "source": metadata.get("source") if metadata else None,  # Source file/reference
                "distance": distance  # Similarity distance score
            })
    
    if not matches and mode == "evidence":
        return {
            "answer": "No encontré evidencia suficiente en el contexto para responder esa pregunta.",
            "matches": []
        }

    # Build response depending on selected mode
    # if mode == "evidence":
    #     answer = build_evidence_answer(matches)
    # elif mode == "persona":
    #     answer = build_persona_answer(matches)

    response = generate_answer(question, matches, mode)

    # Return both the combined context and individual matches
    return {
        "answer": response,
        "matches": matches
    }

def build_evidence_answer(matches):
    """
    Build a simple evidence-based answer.
    Current MVP behavior:
    - Returns the most relevant retrieved passage.
    - Later this can be replaced with an LLM-generated grounded answer.
    """

    if not matches:
        return "No relevant evidence found in the indexed corpus."

    return f"Most relevant evidence found: {matches[0]['text']}"

def build_persona_answer(matches):
    """
    Build a persona-inspired answer.
    Current MVP behavior:
    - Uses the most relevant retrieved passage.
    - Later this can be replaced with an LLM prompt that writes in a
      Steve Jobs-inspired tone while staying grounded in retrieved evidence.
    """

    if not matches:
        return "No relevant evidence found in the indexed corpus."

    return f"Persona-inspired draft based on retrieved evidence: {matches[0]['text']}"

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    """
    Split text into fixed-size character chunks with overlap.
    This was the first chunking strategy.
    It is simple, but it can cut words or ideas in the middle.
    """

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        # Move forward while keeping some overlap with the previous chunk
        start += chunk_size - overlap

    return chunks



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
        "URL",
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