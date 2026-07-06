# Retrieval and response-building logic for the RAG system
import logging

from app.llm import generate_answer


logger = logging.getLogger(__name__)

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
    
    logger.info("Retrieval started. mode=%s top_k=%d question=%r", mode, top_k, question)

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
    ids       = results["ids"][0]

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
    
    logger.info(
        "Distance filter applied. retrieved=%d matched=%d max_distance=%.2f",
        len(documents), len(matches), max_distance
    )

    if not matches and mode == "evidence":
        logger.warning("No matches passed the distance filter. Returning fallback. mode=%s", mode)
        return {
            "answer": "No matches passed the distance filter",
            "matches": []
        }

    # If the selected mode is "persona", the response is generated even if there are no matches
    response = generate_answer(question, matches, mode)

    # Return both the combined context and individual matches
    return {
        "answer": response,
        "matches": matches
    }


