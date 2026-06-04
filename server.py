from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from supabase import create_client
import os

app = FastAPI()

model    = SentenceTransformer("all-MiniLM-L6-v2")
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

class QueryRequest(BaseModel):
    query:     str
    category:  str | None = None
    top_k:     int = 3
    threshold: float = 0.45

@app.post("/retrieve")
async def retrieve_knowledge(req: QueryRequest):
    embedding = model.encode(req.query.replace("\n", " ").strip(), normalize_embeddings=True).tolist()

    params = {"query_embedding": embedding, "match_count": req.top_k}
    if req.category:
        params["filter_category"] = req.category

    result = supabase.rpc("match_sandra_knowledge", params).execute()
    chunks = result.data

    if not chunks:
        return {"context": "", "status": "no_results", "action": "escalate"}

    if chunks[0]["similarity"] < req.threshold:
        return {"context": "", "status": "low_confidence", "action": "escalate"}

    context = "\n\n".join([
        f"[{c['category'].upper()}]\n{c['content']}" for c in chunks
    ])

    return {"context": context, "status": "ok", "action": "answer", "chunks": len(chunks)}

@app.get("/health")
async def health():
    return {"status": "ok", "assistant": "Sandra", "model": "all-MiniLM-L6-v2"}
