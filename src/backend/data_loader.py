from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import os
from pathlib import Path

# Provide robust absolute path to the root .env
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

client = OpenAI(api_key=os.getenv("OPENAI_API"))
EMBED_MODEL = os.getenv("EMBED_MODEL")
EMBED_DIM = int(os.getenv("EMBED_DIM"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))

splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]
