from __future__ import annotations
import os, re, json, asyncio
from typing import List, Optional, Literal

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import google.generativeai as genai

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --------------------------- 1) Config ---------------------------
load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not SERPAPI_API_KEY:
    raise RuntimeError("Missing SERPAPI_API_KEY in environment.")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in environment.")

genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-1.5-flash"

# --------------------------- 2) TechMate system prompt ---------------------------
TECHMATE_SYSTEM_PROMPT = (
    "You are TechMate, an agentic virtual tech support assistant. "
    "Help users resolve any tech issue safely and step-by-step. "
    "Produce structured JSON exactly matching the TechMateOutput schema. "
    "Include OS/device-specific commands, cite sources, and note assumptions. "
    "If unsure or steps fail, propose alternatives or escalate. "
    "Never suggest risky actions without explicit confirmation."
)

# --------------------------- 3) Output schema ---------------------------
class Step(BaseModel):
    id: str
    title: str
    rationale: str
    action: str
    os: Optional[Literal["Windows","macOS","Linux","Any"]] = "Any"
    commands: List[str] = []
    expect: str
    if_fails_next: Optional[str] = None

class TechMateOutput(BaseModel):
    issue_summary: str
    likely_causes: List[str]
    plan_overview: List[str]
    steps: List[Step]
    quick_checks: List[str] = []
    diagnostics_to_collect: List[str] = []
    resolution_criteria: List[str] = []
    escalation_criteria: List[str] = []
    safety_notes: List[str] = []
    sources: List[str] = []
    assumptions: List[str] = []
    confidence: float

# --------------------------- 4) Utilities: search + scrape ---------------------------
SERPAPI_ENDPOINT = "https://serpapi.com/search.json"

async def serpapi_search(query: str, num: int = 5) -> List[dict]:
    params = {"engine": "google","q": query,"num": num,"api_key": SERPAPI_API_KEY}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(SERPAPI_ENDPOINT, params=params)
        r.raise_for_status()
        return r.json().get("organic_results", [])

def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script","style","noscript"]):
        tag.extract()
    text = soup.get_text("\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+"," ",text)
    return text.strip()

async def fetch_page_text(url: str, max_chars: int = 20000) -> str:
    headers = {"User-Agent": "TechMateBot/1.0"}
    async with httpx.AsyncClient(timeout=30, headers=headers, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return clean_text(resp.text)[:max_chars]

# --------------------------- 5) RAG setup ---------------------------
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
vector_index = None
chunk_texts = []

def create_embeddings(chunks: list[str]):
    global vector_index, chunk_texts
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    vector_index = faiss.IndexFlatL2(dim)
    vector_index.add(embeddings)
    chunk_texts = chunks

def retrieve_relevant(query: str, top_k: int = 5) -> list[str]:
    global vector_index, chunk_texts
    if vector_index is None:
        return []
    q_emb = embedding_model.encode([query], convert_to_numpy=True)
    D, I = vector_index.search(q_emb, top_k)
    return [chunk_texts[i] for i in I[0] if i < len(chunk_texts)]

# --------------------------- 6) Gemini call ---------------------------
async def ask_gemini_techmate(user_context: dict, page_snippets: List[dict]) -> TechMateOutput:
    model = genai.GenerativeModel(MODEL_NAME, system_instruction=TECHMATE_SYSTEM_PROMPT)
    prompt = {
        "role": "user",
        "parts": [{
            "text": (
                "Generate a **full troubleshooting plan** for the user's issue. "
                "Output must match TechMateOutput schema exactly.\n\n"
                f"JSON Schema:\n{json.dumps(TechMateOutput.model_json_schema(), indent=2)}\n\n"
                f"User context:\n{json.dumps(user_context, indent=2)}\n\n"
                f"Web snippets:\n{json.dumps(page_snippets, indent=2)}\n\n"
                "Output ONLY valid JSON."
            )
        }]
    }
    generation_config = genai.GenerationConfig(response_mime_type="application/json", temperature=0.3)
    resp = await model.generate_content_async(prompt, generation_config=generation_config)
    return TechMateOutput(**json.loads(resp.text))

# --------------------------- 7) Main agent loop with RAG ---------------------------
async def techmate_agent(query: str, device: str="Windows laptop",
                         os_name: Literal["Windows","macOS","Linux"]="Windows",
                         symptoms: Optional[List[str]]=None,
                         constraints: Optional[List[str]]=None) -> TechMateOutput:
    cache = load_cache()

    # 0) Check cache first
    cached_answer = search_cache(query, cache)
    if cached_answer:
        print("🔍 Using cached result")
        return TechMateOutput(**cached_answer)

    # 1) Search top results
    results = await serpapi_search(query, num=5)
    if not results:
        raise RuntimeError("No search results from SerpAPI.")

    # 2) Fetch and chunk content
    all_chunks = []
    for res in results[:5]:
        url = res.get("link") or res.get("url")
        if not url:
            continue
        try:
            text = await fetch_page_text(url)
        except:
            text = res.get("snippet","")
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        all_chunks.extend(chunks)

    # 3) Build embeddings and retrieve
    create_embeddings(all_chunks)
    retrieved_chunks = retrieve_relevant(query, top_k=5)
    page_snippets = [{"url": "RAG-source", "excerpt": chunk} for chunk in retrieved_chunks]

    # 4) User context
    user_ctx = {
        "query": query,
        "device": device,
        "os": os_name,
        "symptoms": symptoms or [],
        "constraints": constraints or ["Prefer safe, built-in solutions first"]
    }

    # 5) Generate structured plan
    plan = await ask_gemini_techmate(user_ctx, page_snippets)

    # 6) Save to cache
    add_to_cache(query, plan.model_dump(), cache)

    return plan

# --------------------------- 8) Example usage ---------------------------
async def _demo():
    user_query = input("Enter your tech issue (e.g., 'WiFi disconnects after sleep'): ")
    plan = await techmate_agent(query=user_query, device="Dell XPS 13", os_name="Windows")
    print(json.dumps(plan.model_dump(), indent=2))
CACHE_FILE = "techmate_cache.json"

def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {"queries": []}

def save_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def search_cache(query: str, cache: dict) -> Optional[dict]:
    # simple exact match; later we can do embeddings for fuzzy search
    for entry in cache["queries"]:
        if query.lower() == entry["query"].lower():
            return entry["answer"]
    return None

def add_to_cache(query: str, answer: dict, cache: dict):
    cache["queries"].append({"query": query, "answer": answer})
    save_cache(cache)

if __name__ == "__main__":
    asyncio.run(_demo())
