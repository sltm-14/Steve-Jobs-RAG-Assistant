# Steve Jobs RAG Assistant

A RAG-powered assistant that lets you query Steve Jobs' thinking. **evidence mode** returns answers grounded strictly in his documented words, **persona mode** responds in his voice using what's known about him.

## Tech Stack

- **Python**: core language
- **FastAPI**: REST API framework
- **ChromaDB**: vector database for semantic search
- **Sentence Transformers**: text embedding model (all-MiniLM-L6-v2)
- **OpenAI GPT / Anthropic Claude**: LLM generation (configurable via .env)
- **Docker**: containerized for reproducible builds

## Getting Started

### Prerequisites
- Docker and Docker Compose installed
- API key for OpenAI and/or Anthropic

### Setup

1. Clone the repository
   ```bash
   git clone https://github.com/sltm-14/Steve-Jobs-RAG-Assistant.git
   cd sj-rag-assistant
   ```

2. Create a `.env` file in the project root:
   ```
   LLM_PROVIDER=openai        # or "anthropic"
   OPENAI_API_KEY=your-key-here
   ANTHROPIC_API_KEY=your-key-here
   ```

3. Build and start the container:
   ```bash
   docker compose up --build
   ```

4. Open `http://localhost:8000/docs` and try the `/ask` endpoint. You can use the following example:

   ```json
   {
     "question": "What did Steve Jobs think about design?",
     "top_k": 3,
     "mode": "persona"
   }
   ```

### Adding more documents (optional)

Upload additional `.txt` or `.pdf` files through the `/ingest_file` endpoint at `http://localhost:8000/docs`.

### Switching LLM providers

Edit `LLM_PROVIDER` in your `.env` and restart:

```bash
docker compose up
```

No rebuild needed, only configuration changed.

## Technical Decisions

- **Paragraph-based chunking**: splits text at paragraph boundaries instead of fixed character counts, preserving context and avoiding mid-sentence cuts.
- **ChromaDB**: lightweight, open source vector database with disk persistence. No separate server required, well suited for projects of this scale.
- **all-MiniLM-L6-v2**: small, fast embedding model with strong semantic search quality on CPU. No GPU required.
- **Configurable LLM provider**: switch between OpenAI and Anthropic by changing one line in `.env`, no code changes needed.
- **Single model and collection instance**: initialized once in `main.py` and passed as parameters, avoiding duplicated instances and making components easier to test in isolation.
