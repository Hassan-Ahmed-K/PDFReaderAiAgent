import logging
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
import inngest
import inngest.fast_api
from inngest.experimental import ai
import uuid
import os
import sys
from pathlib import Path
import shutil

# ==============================
# PROJECT ROOT + ENV
# ==============================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
sys.path.append(str(PROJECT_ROOT))

# ==============================
# LOCAL IMPORTS
# ==============================
from src.backend.data_loader import load_and_chunk_pdf, embed_texts
from src.backend.qdrant_db import QdrantStorage
from src.backend.schemas import (
    RAGSearchResult,
    RAGUpsertResult,
    RAGChunkAndSrc
)

# ==============================
# QDRANT INIT
# ==============================
qdrant_storage = QdrantStorage(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    dims=int(os.getenv("EMBED_DIM"))
)

# ==============================
# INNGEST CLIENT
# ==============================
inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    serializer=inngest.PydanticSerializer(),
    signing_key=os.getenv("INNGEST_SIGNING_KEY"),
    event_key=os.getenv("INNGEST_EVENT_KEY"),
    is_production=True
)


RESULT_STORE = {}

# ==============================
# FASTAPI APP
# ==============================
app = FastAPI()

@app.get("/")
def root():
    return {"msg": "API running"}

# ==============================
# FILE UPLOAD ENDPOINT
# ==============================
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    uploads_dir = PROJECT_ROOT / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    file_path = uploads_dir / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "file_path": str(file_path),
        "filename": file.filename
    }

@app.get("/api/result/{event_id}")
def get_result(event_id: str):
    return RESULT_STORE.get(event_id, {"status": "processing"})


# ==============================
# INNGEST FUNCTION: INGEST PDF
# ==============================
@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):

    def _load():
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)

        filename = os.path.basename(pdf_path.replace("\\", "/"))
        actual_path = PROJECT_ROOT / "uploads" / filename

        chunks = load_and_chunk_pdf(str(actual_path))
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(data: RAGChunkAndSrc):
        vecs = embed_texts(data.chunks)

        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{data.source_id}:{i}"))
            for i in range(len(data.chunks))
        ]

        payloads = [
            {"source": data.source_id, "text": data.chunks[i]}
            for i in range(len(data.chunks))
        ]

        qdrant_storage.upsert(ids, vecs, payloads)

        return RAGUpsertResult(ingested=len(data.chunks))

    chunks = await ctx.step.run("load", _load, output_type=RAGChunkAndSrc)
    result = await ctx.step.run("upsert", lambda: _upsert(chunks), output_type=RAGUpsertResult)

    return result.model_dump()

# ==============================
# INNGEST FUNCTION: QUERY
# ==============================
@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):

    def _search():
        question = ctx.event.data["question"]
        top_k = int(ctx.event.data.get("top_k", 5))

        query_vec = embed_texts([question])[0]
        found = qdrant_storage.search(query_vec, top_k)

        return RAGSearchResult(**found)

    found = await ctx.step.run(
        "search",
        _search,
        output_type=RAGSearchResult
    )

    context = "\n\n".join(found.contexts)

    prompt = f"""
                Use the context below to answer.

                Context:
                {context}

                Answer the question:
                """

    adapter = ai.openai.Adapter(
        auth_key=os.getenv("OPENAI_API"),
        model="gpt-4o-mini"
    )

    res = await ctx.step.ai.infer(
        "llm",
        adapter=adapter,
        body={
            "messages": [
                {"role": "system", "content": "Answer only from context."},
                {"role": "user", "content": prompt}
            ]
        }
    )

    answer = res["choices"][0]["message"]["content"]

    RESULT_STORE[ctx.event.id] = {
        "answer": answer,
        "sources": found.sources,
        "num_contexts": len(found.contexts)
    }

    return {
        "answer": answer,
        "sources": found.sources,
        "num_contexts": len(found.contexts)
    }

# ==============================
# INNGEST ROUTER (CRITICAL FIX)
inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])

# ==============================
# RUN SERVER
# ==============================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("src.backend.main:app", host="0.0.0.0", port=port)