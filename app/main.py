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
    # Health check endpoint
    return {"message": "Steve Jobs RAG Assistant is running"}


@app.post("/ingest_file")
async def upload_file(uploaded_file: UploadFile):
    content = await uploaded_file.read()
    text = content.decode("utf-8")

    print("filename:", uploaded_file.filename)
    print("text length:", len(text))
    print("text preview:", repr(text[:200]))

    return ingest_file(text, uploaded_file.filename)


@app.post("/ask")
def ask(request: AskRequest):

    retrieval_result = run_retrieval(
        question=request.question,
        top_k=request.top_k,
        mode = request.mode,
        model=model,
        collection=collection
    )

    return {
        "question": request.question,
        "mode": request.mode,
        "answer": retrieval_result["answer"],
        "matches": retrieval_result["matches"]
    }
