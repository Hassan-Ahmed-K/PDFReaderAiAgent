import requests

requests.post(
    "https://api.inngest.com/e/YOUR_EVENT_KEY",
    json={
        "name": "rag/ingest_pdf",
        "data": {"msg": "hello from prod"}
    }
)






