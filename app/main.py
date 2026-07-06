import logging

from fastapi import FastAPI, UploadFile
from sentence_transformers import SentenceTransformer
import chromadb

from app.rag import run_retrieval
from app.ingest import ingest_file
from app.ingest import extract_text_from_pdf
from app.schemas import AskRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(title="Steve Jobs RAG Assistant")

# Load embedding model for transforming text into vectors
# This model converts text chunks into embedding vectors.
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

    filename = uploaded_file.filename or "uploaded_file"

    if filename.lower().endswith(".txt"):
        text = content.decode("utf-8")

    elif filename.lower().endswith(".pdf"):
        text = extract_text_from_pdf(content)

    else:
        logger.warning("Unsupported file type received. filename=%s", filename)
        return {
            "error": "Unsupported file type. Please upload a .txt or .pdf file."
        }

    logger.info("File received. filename=%s size=%d bytes", filename, len(content))
    logger.debug("Text preview: %s", repr(text[:200]))

    # Send text to ingestion pipeline
    return ingest_file(text, uploaded_file.filename, model)



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