import asyncio
import time
import os
import requests
import streamlit as st
import inngest
from dotenv import load_dotenv

# -------------------------------
# CONFIG
# -------------------------------
def get_secret(key, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)


load_dotenv()

INNGEST_EVENT_KEY = get_secret("INNGEST_EVENT_KEY") 
INNGEST_REST_API_KEY =  get_secret("INNGEST_REST_API_KEY") 
INNGEST_API_BASE = get_secret("INNGEST_API_BASE")
FASTAPI_URL = get_secret("FAST_API_URL")

print("INNGEST_EVENT_KEY", INNGEST_EVENT_KEY)
print("INNGEST_REST_API_KEY", INNGEST_REST_API_KEY)
print("INNGEST_API_BASE", INNGEST_API_BASE)
print("FAST_API_URL", FASTAPI_URL)

# -------------------------------
# STREAMLIT SETUP
# -------------------------------
st.set_page_config(page_title="RAG PDF App", page_icon="📄", layout="centered")

# -------------------------------
# 🟢 BACKEND LOADER (RUN ONCE, 90s)
# -------------------------------
def wait_for_backend(timeout=90):
    with st.spinner("🚀 Waking up AI server..."):
        start = time.time()

        while True:
            try:
                res = requests.get(FASTAPI_URL, timeout=10)

                if res.status_code == 200 and res.json().get("msg") == "API running":
                    return True

            except Exception:
                pass

            if time.time() - start > timeout:
                st.error("❌ Backend is not responding (Render cold start issue)")
                st.stop()

            time.sleep(2)


if "backend_ready" not in st.session_state:
    wait_for_backend()
    st.session_state.backend_ready = True

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
    except:
        return None


# -------------------------------
# 🎨 UI HEADER
# -------------------------------
st.markdown("""
# 🤖 PDF RAG AI Assistant  
Upload PDF → Ask Questions → Get AI Answers
""")

# -------------------------------
# 📤 UPLOAD SECTION
# -------------------------------
st.subheader("📤 Upload PDF")

file = st.file_uploader("Drop your PDF here", type=["pdf"])

if file:
    with st.spinner("Uploading & indexing..."):
        file_path, filename = upload_to_backend(file)
        event_id = asyncio.run(send_ingest(file_path, filename))

    st.success("PDF uploaded successfully")
    st.caption(f"Ingest Event ID: {event_id}")

st.divider()

# -------------------------------
# 💬 QUERY SECTION
# -------------------------------
st.subheader("💬 Ask Questions")

question = st.text_input("Ask something from your PDF")
top_k = st.slider("Context Depth (Top K)", 1, 20, 5)

if st.button("🚀 Ask AI") and question.strip():

    with st.spinner("Sending query to AI..."):
        event_id = asyncio.run(send_query(question.strip(), int(top_k)))

    st.caption(f"Event ID: {event_id}")

    # ---------------------------
    # 🔥 RESULT POLLING (90 sec)
    # ---------------------------
    placeholder = st.empty()
    start = time.time()

    while True:
        result = get_result(event_id)

        if result and result.get("answer"):
            placeholder.empty()

            st.success("✅ Answer Ready")

            st.markdown("### 🧠 Answer")
            st.write(result["answer"])

            sources = result.get("sources", [])
            if sources:
                st.markdown("### 📚 Sources")
                for s in sources:
                    st.write(f"- {s}")

            break

        # animated loader
        dots = "." * (int(time.time() * 2) % 4)
        placeholder.info(f"🤖 Thinking{dots}")

        time.sleep(1)

        # timeout
        if time.time() - start > 90:
            placeholder.error("⚠️ Timeout: AI is taking too long")
            break