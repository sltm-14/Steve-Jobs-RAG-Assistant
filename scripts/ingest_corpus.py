from app.ingest import ingest_from_path

result = ingest_from_path(
    "data/raw/make_something_wonderful_full.txt",
    "make_something_wonderful_full.txt | Steve Jobs Archive"
)

print(result)