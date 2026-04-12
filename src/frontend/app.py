import asyncio
import time
import os
import requests
import streamlit as st
import inngest

# -------------------------------
# CONFIG
# -------------------------------
def get_secret(key, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

INNGEST_EVENT_KEY = get_secret("INNGEST_EVENT_KEY","uTah3eSR2Mi5RA0oBI4EmfLj34qO-Wk0Q0XCJyQaZ3ExMXDCMOfQ3R-4qFO4NuJGOo5ZsSkWzWOsTJPnl7t8FQ") 
INNGEST_REST_API_KEY = get_secret("INNGEST_REST_API_KEY","eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6OTk1NmE2OGYtZTQ3ZC00ZWI5LWJmZDctMDdhYmMyZjllMTZkIn0.6jqqm-GcYlt3uAe88tHvQSTSpSTjKguR_XCx3LvaYOU") 
INNGEST_API_BASE = get_secret("INNGEST_API_BASE", "https://api.inngest.com/v1")
FASTAPI_URL = get_secret("FASTAPI_URL","https://pdfreaderaiagent.onrender.com")

# -------------------------------
# STREAMLIT SETUP
# -------------------------------
st.set_page_config(page_title="RAG PDF App", page_icon="📄")

# -------------------------------
# INNGEST CLIENT
# -------------------------------
@st.cache_resource
def get_client():
    if not INNGEST_EVENT_KEY:
        raise ValueError("Missing INNGEST_EVENT_KEY")

    return inngest.Inngest(
        app_id="rag_app",
        event_key=INNGEST_EVENT_KEY,
        is_production=True,
    )

# -------------------------------
# UPLOAD FILE
# -------------------------------
def upload_to_backend(file):
    files = {"file": (file.name, file.getvalue(), "application/pdf")}
    response = requests.post(f"{FASTAPI_URL}/api/upload", files=files)

    response.raise_for_status()
    data = response.json()

    return data["file_path"], data["filename"]

# -------------------------------
# SEND INGEST EVENT
# -------------------------------
async def send_ingest(file_path, filename):
    client = get_client()

    result = await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": file_path,
                "source_id": filename,
            },
        )
    )

    return result[0]

# -------------------------------
# SEND QUERY EVENT
# -------------------------------
async def send_query(question, top_k):
    client = get_client()

    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )

    return result[0]

# -------------------------------
# GET RESULT FROM FASTAPI
# -------------------------------
def get_result(event_id):
    try:
        res = requests.get(f"{FASTAPI_URL}/api/result/{event_id}")

        if res.status_code == 200:
            return res.json()

        return None

    except Exception:
        return None

# -------------------------------
# UI
# -------------------------------
st.title("📄 PDF RAG System")

# ===============================
# UPLOAD SECTION
# ===============================
file = st.file_uploader("Upload PDF", type=["pdf"])

if file:
    with st.spinner("Uploading file..."):
        file_path, filename = upload_to_backend(file)

    with st.spinner("Sending ingestion event..."):
        event_id = asyncio.run(send_ingest(file_path, filename))

    st.success("PDF sent for processing")
    st.info(f"Ingest Event ID: {event_id}")

st.divider()

# ===============================
# QUESTION SECTION
# ===============================
st.title("💬 Ask Questions")

with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("Top K", min_value=1, max_value=20, value=5)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():

        with st.spinner("Sending query..."):
            event_id = asyncio.run(send_query(question.strip(), int(top_k)))

        st.success("Query sent")
        st.info(f"Event ID: {event_id}")

        # ---------------------------
        # 🔥 POLLING FASTAPI RESULT
        # ---------------------------
        placeholder = st.empty()

        for i in range(60):
            result = get_result(event_id)

            if result and result.get("answer"):
                placeholder.success(result["answer"])

                sources = result.get("sources", [])
                if sources:
                    st.subheader("Sources")
                    for s in sources:
                        st.write(f"- {s}")

                break

            placeholder.info(f"Processing... {i+1}/60")
            time.sleep(1)

        else:
            placeholder.warning("Still processing... please try again.")