import os
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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

# Must match ingestion embedding model!
PINECONE_EMBED_MODEL = os.environ.get("PINECONE_EMBED_MODEL", "llama-text-embed-v2")

# LLM model for generation
CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")


if not PINECONE_API_KEY:
    raise RuntimeError("Missing env var PINECONE_API_KEY")

# OpenAI is required only for /generate (search-only still works without it)
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# -------------------------
# Clients
# -------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


# -------------------------
# FastAPI + Static UI
# -------------------------
app = FastAPI(title="RAG Policy Agent (Pinecone embeddings + OpenAI LLM)")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home():
    return FileResponse("static/index.html")


# -------------------------
# Schemas
# -------------------------
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    doc_id: Optional[str] = None
    top_k: int = 5


class Match(BaseModel):
    id: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    matches: List[Match]


class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=1)
    doc_id: Optional[str] = None
    top_k: int = 5


class GenerateResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]


# -------------------------
# Helpers
# -------------------------
def is_greeting(q: str) -> bool:
    qn = q.strip().lower()
    return qn in {"hi", "hello", "hey", "hii", "hiii", "yo", "good morning", "good evening"}


def embed_query_with_pinecone(query: str) -> List[float]:
    """
    Pinecone-hosted embedding model. No local Ollama required.
    Must match ingestion model/dimensions.
    """
    emb = pc.inference.embed(
        model=PINECONE_EMBED_MODEL,
        inputs=[query],
        parameters={"input_type": "query"},
    )
    return emb.data[0].values


def pinecone_search(query: str, doc_id: Optional[str], top_k: int):
    qvec = embed_query_with_pinecone(query)

    flt = None
    if doc_id:
        # requires your metadata to include doc_id
        flt = {"doc_id": {"$eq": doc_id}}

    res = index.query(
        vector=qvec,
        top_k=top_k,
        include_metadata=True,
        namespace=NAMESPACE,
        filter=flt,
    )
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
        doc_id = md.get("doc_id")
        file_name = md.get("file_name")
        source_type = md.get("source_type")
        chunk_index = md.get("chunk_index")

        header = f"[doc_id={doc_id} file={file_name} type={source_type} chunk={chunk_index}]"

        snippet = (chunk_text[:1200] + "…") if len(chunk_text) > 1200 else chunk_text
        block = header + "\n" + snippet

        if used + len(block) > max_chars:
            break

        parts.append(block)
        used += len(block)

    return "\n\n---\n\n".join(parts)


def llm_generate_answer(query: str, context: str) -> str:
    if openai_client is None:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY not set; cannot generate answers.",
        )

    system = (
        "You are a helpful assistant for policy/SOP Q&A. "
        "Use ONLY the provided context. "
        "If the answer is not in the context, say: "
        "\"I don't know based on the provided documents.\" "
        "Be concise. Use bullets when helpful."
    )

    prompt = f"""CONTEXT:
{context}

QUESTION:
{query}
"""

    resp = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


# -------------------------
# Endpoints
# -------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "index": INDEX_NAME,
        "namespace": NAMESPACE,
        "embed_model": PINECONE_EMBED_MODEL,
        "llm_model": CHAT_MODEL,
        "openai_enabled": bool(OPENAI_API_KEY),
    }


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    res = pinecone_search(req.query, req.doc_id, req.top_k)

    matches_out = []
    for m in res.get("matches", []):
        matches_out.append(
            Match(
                id=m["id"],
                score=float(m["score"]),
                metadata=m.get("metadata") or {},
            )
        )
    return SearchResponse(matches=matches_out)


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    # Friendly fallback for greetings
    if is_greeting(req.query):
        return GenerateResponse(
            answer="Hi! 👋 Ask me a question about the documents (policies/SOPs), and I’ll answer using the ingested content.",
            citations=[],
        )

    res = pinecone_search(req.query, req.doc_id, req.top_k)
    matches = res.get("matches", [])

    if not matches:
        return GenerateResponse(
            answer="I couldn't find relevant information in the documents.",
            citations=[],
        )

    context = build_context_from_matches(matches)
    answer = llm_generate_answer(req.query, context)

    citations = []
    for m in matches:
        md = m.get("metadata") or {}
        citations.append(
            {
                "id": m.get("id"),
                "score": float(m.get("score", 0.0)),
                "doc_id": md.get("doc_id"),
                "file_name": md.get("file_name"),
                "source_type": md.get("source_type"),
                "chunk_index": md.get("chunk_index"),
            }
        )

    return GenerateResponse(answer=answer, citations=citations)
