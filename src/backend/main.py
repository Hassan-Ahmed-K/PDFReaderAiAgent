import logging
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
import inngest
import inngest.fast_api
import uuid
import os
import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")
sys.path.append(str(PROJECT_ROOT))

from src.backend.data_loader import load_and_chunk_pdf, embed_texts
from src.backend.qdrant_db import QdrantStorage
from src.backend.schemas import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult

# -------------------------------
# INIT
# -------------------------------
qdrant_storage = QdrantStorage(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    dims=int(os.getenv("EMBED_DIM"))
)

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    signing_key=os.getenv("INNGEST_SIGNING_KEY"),
    event_key=os.getenv("INNGEST_EVENT_KEY"),
    is_production=True,
)

# -------------------------------
# INGEST FUNCTION
# -------------------------------
@inngest_client.create_function(
    fn_id="RAG Ingest",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def ingest(ctx: inngest.Context):

    def _load(ctx):
        pdf_path = ctx.event.data["pdf_path"]

        filename = os.path.basename(pdf_path.replace("\\", "/"))
        actual_path = PROJECT_ROOT / "uploads" / filename

        if not actual_path.exists():
            raise FileNotFoundError(f"{actual_path} not found")

        chunks = load_and_chunk_pdf(str(actual_path))
        return RAGChunkAndSrc(chunks=chunks, source_id=filename)

    def _upsert(data):
        vecs = embed_texts(data.chunks)

        ids = [str(uuid.uuid4()) for _ in data.chunks]
        payloads = [{"text": c, "source": data.source_id} for c in data.chunks]

        qdrant_storage.upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(data.chunks))

    data = await ctx.step.run("load", lambda: _load(ctx))
    result = await ctx.step.run("upsert", lambda: _upsert(data))

    return result.model_dump()

# -------------------------------
# QUERY FUNCTION
# -------------------------------
@inngest_client.create_function(
    fn_id="RAG Query",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def query(ctx: inngest.Context):

    question = ctx.event.data["question"]
    top_k = ctx.event.data.get("top_k", 5)

    def _search():
        vec = embed_texts([question])[0]
        return qdrant_storage.search(vec, top_k)

    found = await ctx.step.run("search", _search)

    return {"answer": str(found)}

# -------------------------------
# FASTAPI APP
# -------------------------------
app = FastAPI()

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    uploads = PROJECT_ROOT / "uploads"
    uploads.mkdir(exist_ok=True)

    path = uploads / file.filename

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"file_path": str(path)}  # ✅ IMPORTANT

# -------------------------------
# INNGEST SERVE
# -------------------------------
inngest.fast_api.serve(app, inngest_client, [ingest, query])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.backend.main:app", host="0.0.0.0", port=8000)