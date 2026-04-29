from fastapi import FastAPI, UploadFile
from sentence_transformers import SentenceTransformer
from app.rag import run_retrieval
from app.ingest import ingest_file
import chromadb

from app.schemas import AskRequest

# Initialize FastAPI application
app = FastAPI(title="Steve Jobs RAG Assistant")

# Load embedding model for transforming text into vectors
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create or connect to persistent ChromaDB storage
client = chromadb.PersistentClient(path="./chroma_db")

# Create or get the collection where documents will be stored
collection = client.get_or_create_collection(name="steve_jobs_corpus")


@app.get("/")

def root():
    """
    Basic health check endpoint to verify that the API is running.
    """
    return {"message": "Steve Jobs RAG Assistant is running"}



@app.post("/ingest_file")

async def upload_file(uploaded_file: UploadFile):
    """
    Upload a .txt file and ingest it into the vector database.
    Steps:

    1. Read uploaded file bytes
    2. Decode bytes into text
    3. Clean text
    4. Chunk text
    5. Generate embeddings
    6. Store in ChromaDB
    """

    # Read raw uploaded file content
    content = await uploaded_file.read()

    # Convert bytes into utf-8 text
    text = content.decode("utf-8")

    # Debug info in terminal
    print("filename:", uploaded_file.filename)
    print("text length:", len(text))
    print("text preview:", repr(text[:200]))

    # Send text to ingestion pipeline
    return ingest_file(text, uploaded_file.filename)



@app.post("/ask")
def ask(request: AskRequest):
    """
    Query the indexed corpus.
    Request body includes:
    - question: user question
    - top_k: number of chunks to retrieve
    - mode: evidence or persona
    """

    # Run semantic retrieval pipeline
    retrieval_result = run_retrieval(
        question=request.question,
        top_k=request.top_k,
        mode=request.mode,
        model=model,
        collection=collection
    )

    sources = []
    for match in retrieval_result["matches"]:
        sources.append({
            "source": match["source"],
            "chunk_id": match["id"],
            "distance": match["distance"]
        })

    # Return final response
    return {
        "question": request.question,
        "mode": request.mode,
        "answer": retrieval_result["answer"],
        "sources": sources,
        "matches": retrieval_result["matches"]
    }