# Retrieval and responsese logic
# Convert user question into embedding vector

def run_retrieval(question: str, top_k: int, mode: str, model, collection, max_distance: float = 0.85):
    
    if mode not in ("evidence", "persona"):
        raise ValueError("mode debe ser 'evidence' o 'persona'")
    
    # Generate the embedding vector for the input question
    query_embedding = model.encode([question]).tolist()

    # Search the vector database and return the top_k closest matches
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    # Extract returned fields from the query results
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    # Store filtered matches here
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

    # Combine all matched document texts into a single context string
    #context = "\n\n".join([m["text"] for m in matches])

    if mode == "evidence":

        answer = build_evidence_answer(matches)

    elif mode == "persona":

        answer = build_persona_answer(matches)


    # Return both the combined context and individual matches
    return {
        "answer": answer,
        "matches": matches
    }


def build_evidence_answer(matches):

    if not matches:
        return "No relevant evidence found in the indexed corpus."

    answer = f"Most relevant evidence found: {matches[0]['text']}"

    return answer


def build_persona_answer(matches):

    if not matches:
        return "No relevant evidence found in the indexed corpus."

    answer = f"Persona-inspired draft based on retrieved evidence: {matches[0]['text']}"

    return answer


def chunk_text(text:str, chunk_size: int = 1000, overlap: int = 200):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        
        chunk = text[start:end]
        chunks.append(chunk)

        start += chunk_size - overlap
        
    return chunks


def chunk_text_by_paragraphs(text: str, max_chars: int = 1000):
    chunks = []
    lines = text.splitlines(keepends=True)

    chunk = ""

    for line in lines:
        # If a single line is longer than max_chars, store current chunk first
        if len(line) > max_chars:
            if chunk:
                chunks.append(chunk)
                chunk = ""
            chunks.append(line)
            continue

        if len(chunk) + len(line) <= max_chars:
            chunk += line
        else:
            if chunk:
                chunks.append(chunk)
            chunk = line

    if chunk:
        chunks.append(chunk)

    return chunks


def clean_text(text: str) -> str:
    lines = text.splitlines()
    forbidden_words = ["Copyright", "Typeset", "Credits", "URL", "Published"]

    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if len(line) <= 14:
            continue

        if any(word in line for word in forbidden_words):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

