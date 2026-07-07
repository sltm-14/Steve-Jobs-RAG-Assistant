import chromadb

from app.ingest import ingest_from_path
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Create or connect to persistent ChromaDB storage
client = chromadb.PersistentClient(path="./chroma_db")

# Create or get the collection where documents will be stored
collection = client.get_or_create_collection(name="steve_jobs_corpus")

result = ingest_from_path(
    "data/raw/make_something_wonderful_full.txt",
    "make_something_wonderful_full.txt | Steve Jobs Archive",model
)

print(result)