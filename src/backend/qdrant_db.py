import time
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from qdrant_client.http.exceptions import ResponseHandlingException

class QdrantStorage:
    def __init__(self, url="http://localhost:6333",api_key="", collection="docs", dims=3072):
        if(api_key != ""):
            self.client = QdrantClient(url=url, api_key=api_key, timeout=60)
        else:
            self.client = QdrantClient(url=url, timeout=60)
            
        self.collection = collection

        if not self.client.collection_exists(self.collection):
            for _ in range(5):
                try:
                    self.client.create_collection(
                        collection_name=self.collection,
                        vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
                    )
                    break
                except ResponseHandlingException:
                    time.sleep(5)

    def upsert(self, ids, vector, payload):
        points = [
            PointStruct(
                id=id,
                vector=vector,
                payload=payload,
            )
            for id, vector, payload in zip(ids, vector, payload)
        ]

        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector, top_k:int=5):
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k,
        )
        contexts =[]
        sources = set()

        for r in results.points:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text")
            source = payload.get("source")
            if text:
                contexts.append(text)
                sources.add(source)

        return  {"contexts": contexts, "sources": list(sources)}
