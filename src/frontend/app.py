import asyncio
import time
from pathlib import Path
import requests
import streamlit as st
import inngest
import os

# -------------------------------
# CONFIG (Secrets)
# -------------------------------
def get_secret(key, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

INNGEST_EVENT_KEY = get_secret("INNGEST_EVENT_KEY")
INNGEST_REST_API_KEY = get_secret("INNGEST_REST_API_KEY")
INNGEST_API_BASE = get_secret("INNGEST_API_BASE", "https://api.inngest.com/v1")
FASTAPI_URL = get_secret("FASTAPI_URL")

# -------------------------------
# STREAMLIT SETUP
# -------------------------------
st.set_page_config(page_title="RAG PDF App", page_icon="📄")

@st.cache_resource
def get_client():
    return inngest.Inngest(
        app_id="rag_app",
        event_key=INNGEST_EVENT_KEY,
        is_production=True,
    )

# -------------------------------
# UPLOAD TO BACKEND
# -------------------------------
def upload_to_backend(file):
    files = {"file": (file.name, file.getvalue(), "application/pdf")}
    response = requests.post(f"{FASTAPI_URL}/api/upload", files=files)
    response.raise_for_status()

    return response.json()["file_path"]  # ✅ IMPORTANT

# -------------------------------
# SEND INGEST EVENT
# -------------------------------
async def send_ingest(path, filename):
    client = get_client()

    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": path,          # ✅ SERVER PATH
                "source_id": filename,
            },
        )
    )

# -------------------------------
# UI - UPLOAD
# -------------------------------
st.title("📄 Upload PDF")

file = st.file_uploader("Upload PDF", type=["pdf"])

if file:
    with st.spinner("Uploading..."):
        backend_path = upload_to_backend(file)
        asyncio.run(send_ingest(backend_path, file.name))
        time.sleep(0.3)

    st.success("✅ Ingest triggered")

# -------------------------------
# QUERY EVENT
# -------------------------------
async def send_query(question, top_k):
    client = get_client()

    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={"question": question, "top_k": top_k},
        )
    )

    return result[0]

def fetch_runs(event_id):
    url = f"{INNGEST_API_BASE}/events/{event_id}/runs"
    headers = {"Authorization": f"Bearer {INNGEST_REST_API_KEY}"}

    res = requests.get(url, headers=headers)
    res.raise_for_status()
    return res.json().get("data", [])

def wait_for_output(event_id):
    start = time.time()

    while True:
        runs = fetch_runs(event_id)

        if runs:
            run = runs[0]
            status = run.get("status")

            if status in ("Completed", "Succeeded"):
                return run.get("output", {})

        if time.time() - start > 120:
            raise TimeoutError("Timeout waiting for response")

        time.sleep(0.5)

# -------------------------------
# UI - QUERY
# -------------------------------
st.title("💬 Ask Question")

q = st.text_input("Question")
k = st.number_input("Top K", 1, 10, 5)

if st.button("Ask") and q:
    with st.spinner("Thinking..."):
        eid = asyncio.run(send_query(q, int(k)))
        output = wait_for_output(eid)

        st.write(output.get("answer", "No answer"))