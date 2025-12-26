import os
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pinecone import Pinecone
from openai import OpenAI


# -------------------------
# Config (env vars)
# -------------------------
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "test-agent")
NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "toy-agent")

# Pinecone embedding model used during ingestion (must match!)
PINECONE_EMBED_MODEL = os.environ.get("PINECONE_EMBED_MODEL", "llama-text-embed-v2")

# LLM model for generation
CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")


# -------------------------
# Clients
# -------------------------
if not PINECONE_API_KEY:
    raise RuntimeError("Missing env var PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# -------------------------
# FastAPI
# -------------------------
app = FastAPI(title="Toy RAG Agent")


# -------------------------
# Schemas
# -------------------------
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    drug_id: Optional[str] = None
    top_k: int = 5


class Match(BaseModel):
    id: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    matches: List[Match]


class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=1)
    drug_id: Optional[str] = None
    top_k: int = 5


class GenerateResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]


# -------------------------
# Helpers
# -------------------------
def embed_query_with_pinecone(query: str) -> List[float]:
    """
    Use Pinecone Inference to embed query.
    Must match ingestion embedding model & dimensions.
    """
    emb = pc.inference.embed(
        model=PINECONE_EMBED_MODEL,
        inputs=[query],
        parameters={"input_type": "query"},
    )
    return emb.data[0].values


def pinecone_search(query: str, drug_id: Optional[str], top_k: int):
    qvec = embed_query_with_pinecone(query)

    flt = None
    if drug_id:
        flt = {"drug_id": {"$eq": drug_id}}

    res = index.query(
        vector=qvec,
        top_k=top_k,
        include_metadata=True,
        namespace=NAMESPACE,
        filter=flt
    )
    # res is dict-like with "matches"
    return res


def build_context_from_matches(matches, max_chars: int = 6000) -> str:
    """
    Build a context string from retrieved chunks.
    """
    parts = []
    used = 0
    for m in matches:
        md = (m.get("metadata") or {})
        chunk_text = md.get("text") or ""
        source_files = md.get("source_files") or []
        drug_id = md.get("drug_id")
        doc_type = md.get("doc_type")
        header = f"[drug_id={drug_id} doc_type={doc_type} source_files={source_files}]"

        snippet = (chunk_text[:1200] + "…") if len(chunk_text) > 1200 else chunk_text
        block = header + "\n" + snippet

        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)

    return "\n\n---\n\n".join(parts)


def llm_generate_answer(query: str, context: str) -> str:
    if openai_client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set; cannot generate answer.")

    system = (
        "You are a helpful assistant. Answer using ONLY the provided context. "
        "If the context does not contain the answer, say you don't know."
    )

    user = f"""Context:
{context}

User question:
{query}

Return a concise answer."""
    resp = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


# -------------------------
# Endpoints
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok", "index": INDEX_NAME, "namespace": NAMESPACE, "embed_model": PINECONE_EMBED_MODEL}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    res = pinecone_search(req.query, req.drug_id, req.top_k)
    matches_out = []
    for m in res.get("matches", []):
        matches_out.append(Match(id=m["id"], score=float(m["score"]), metadata=m.get("metadata") or {}))
    return SearchResponse(matches=matches_out)


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    res = pinecone_search(req.query, req.drug_id, req.top_k)
    matches = res.get("matches", [])

    if not matches:
        return GenerateResponse(answer="I couldn't find relevant context in the vector database.", citations=[])

    context = build_context_from_matches(matches)

    answer = llm_generate_answer(req.query, context)

    # citations: return match metadata + id/score (simple)
    citations = []
    for m in matches:
        citations.append({
            "id": m["id"],
            "score": float(m["score"]),
            "metadata": m.get("metadata") or {}
        })

    return GenerateResponse(answer=answer, citations=citations)
