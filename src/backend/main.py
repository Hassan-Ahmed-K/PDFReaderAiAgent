import logging
from dotenv import load_dotenv
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
import uuid
import os
import datetime
import sys
from pathlib import Path

# ==========================================
# VERY IMPORTANT: This block must execute BEFORE importing anything from `src.`
# Provide absolute path to root .env and inject PROJECT_ROOT into Python's path so it can find "src".
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
sys.path.append(str(PROJECT_ROOT))

# Now we can safely import from src
from src.backend.data_loader import load_and_chunk_pdf, embed_texts
from src.backend.qdrant_db import QdrantStorage
from src.backend.schemas import RAQQueryResult, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc

qdrant_storage = QdrantStorage(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"), dims=int(os.getenv("EMBED_DIM")))

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        qdrant_storage.upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int=5):
        query_vec = embed_texts([question])[0]
        found = qdrant_storage.search(query_vec, top_k)
        return RAGSearchResult(**found)

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    adapter = ai.openai.Adapter(
        auth_key=os.getenv("OPENAI_API"),
        model="gpt-4o-mini"
    )

    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You answer questions using only the provided context."},
                {"role": "user", "content": user_content}
            ]
        }
    )

    answer = res["choices"][0]["message"]["content"].strip()
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}
        
    
    # result = await ctx.step.run("generate-answer", lambda: _generate(found), output_type=RAQQueryResult)
    result = await ctx.step.run(
                                "generate-answer",
                                _generate,   
                                found,       
                                output_type=RAQQueryResult
                            )
    return result.model_dump()

app = FastAPI()

@app.get("/")
def read_root():
    return {"msg": "API running"}
    
inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])

if __name__ == "__main__":
    import uvicorn
    # Render assigns a dynamic port via the PORT environment variable. 
    # This automatically catches it and binds to 0.0.0.0
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("src.backend.main:app", host="0.0.0.0", port=port)
