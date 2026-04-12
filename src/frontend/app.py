import asyncio
import time
from pathlib import Path
import requests
import streamlit as st
import inngest
import os

# -------------------------------
# CONFIG (Secrets / Env)
# -------------------------------
def get_secret(key: str, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)


INNGEST_EVENT_KEY = get_secret("INNGEST_EVENT_KEY")
INNGEST_REST_API_KEY = get_secret("INNGEST_REST_API_KEY")
INNGEST_API_BASE = get_secret("INNGEST_API_BASE", "https://api.inngest.com/v1")
FASTAPI_URL = get_secret("FAST_API_URL","https://pdfreaderaiagent.onrender.com")

# -------------------------------
# STREAMLIT CONFIG
# -------------------------------
st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")

# -------------------------------
# INNGEST CLIENT
# -------------------------------
@st.cache_resource
def get_inngest_client():
    return inngest.Inngest(
        app_id="rag_app",
        event_key=INNGEST_EVENT_KEY,
        is_production=True,
    )

# -------------------------------
# FILE UPLOAD → FASTAPI
# -------------------------------
def upload_to_backend(file) -> str:
    if not FASTAPI_URL:
        raise ValueError("FASTAPI_URL is not set in secrets")

    files = {"file": (file.name, file.getvalue(), "application/pdf")}

    response = requests.post(f"{FASTAPI_URL}/api/upload", files=files)
    response.raise_for_status()

    # backend must return {"file_path": "..."}
    return response.json()["file_path"]

# -------------------------------
# SEND INGEST EVENT
# -------------------------------
async def send_rag_ingest_event(pdf_path: str):
    client = get_inngest_client()

    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": pdf_path,
                "source_id": Path(pdf_path).name,
            },
        )
    )

# -------------------------------
# UI: UPLOAD PDF
# -------------------------------
st.title("📄 Upload a PDF to Ingest")

uploaded = st.file_uploader("Choose a PDF", type=["pdf"])

if uploaded is not None:
    with st.spinner("Uploading and triggering ingestion..."):
        try:
            backend_path = upload_to_backend(uploaded)
            asyncio.run(send_rag_ingest_event(backend_path))
            time.sleep(0.3)

            st.success(f"✅ Ingestion triggered for: {uploaded.name}")

        except Exception as e:
            st.error(f"❌ Error: {e}")

st.divider()

# -------------------------------
# QUERY EVENT
# -------------------------------
async def send_rag_query_event(question: str, top_k: int):
    client = get_inngest_client()

    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )

    return result[0]  # event_id

# -------------------------------
# FETCH RUNS FROM INNGEST
# -------------------------------
def fetch_runs(event_id: str):
    url = f"{INNGEST_API_BASE}/events/{event_id}/runs"

    headers = {}
    if INNGEST_REST_API_KEY:
        headers["Authorization"] = f"Bearer {INNGEST_REST_API_KEY}"

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()

    return resp.json().get("data", [])

# -------------------------------
# WAIT FOR OUTPUT
# -------------------------------
def wait_for_run_output(event_id: str, timeout_s=120, poll_interval=0.5):
    start = time.time()
    last_status = None

    while True:
        runs = fetch_runs(event_id)

        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status

            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}

            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Run {status}")

        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timeout (last status: {last_status})")

        time.sleep(poll_interval)

# -------------------------------
# UI: QUERY
# -------------------------------
st.title("💬 Ask a Question")

with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("Chunks to retrieve", min_value=1, max_value=20, value=5)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Thinking..."):
            try:
                event_id = asyncio.run(
                    send_rag_query_event(question.strip(), int(top_k))
                )

                output = wait_for_run_output(event_id)

                answer = output.get("answer", "")
                sources = output.get("sources", [])

                st.subheader("🧠 Answer")
                st.write(answer or "(No answer)")

                if sources:
                    st.caption("📚 Sources")
                    for s in sources:
                        st.write(f"- {s}")

            except Exception as e:
                st.error(f"❌ Error: {e}")