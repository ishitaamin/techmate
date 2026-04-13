# backend/agent/techmate_agent.py
from __future__ import annotations
import os
import re
import json
import logging
from typing import List, Optional, Literal

from pydantic import BaseModel, Field
from dotenv import load_dotenv
import google.generativeai as genai
from tavily import AsyncTavilyClient

# Import our Object-Oriented Vector Store
from backend.database.vector_store import VectorStore

# --------------------------- 1) Config & Logging ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("techmate")

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not TAVILY_API_KEY or not GEMINI_API_KEY:
    raise RuntimeError("Missing API Keys (TAVILY_API_KEY or GEMINI_API_KEY) in environment.")

genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash"
tavily_client = AsyncTavilyClient(api_key=TAVILY_API_KEY)

# Initialize our Vector DB instance
vector_store = VectorStore(index_path="data/faiss.index", chunks_path="data/faiss_chunks.json")

# --------------------------- 2) Prompts & Schemas ---------------------------
TECHMATE_SYSTEM_PROMPT = (
    """You are TechMate, an elite Tier-3 IT support specialist and autonomous troubleshooting agent. 
    Your ultimate goal is to resolve the user's hardware and software issues as quickly and practically as possible.

    CORE RULES OF ENGAGEMENT:
    1. Bias for Action: NEVER ask open-ended diagnostic questions (e.g., "Where is the noise coming from?"). Instead, provide concrete, actionable steps the user can execute immediately (e.g., "Put your ear near the left vent to check if the fan is hitting a wire").
    2. Progressive Troubleshooting: Prioritize the most common, safest, and easiest fixes in the first steps (e.g., restarts, cleaning, basic settings). Reserve complex software registry edits or invasive hardware checks for later steps.
    3. Empathy & Clarity: Briefly acknowledge the user's specific device and issue in the summary. Keep your tone reassuring, professional, and confident.
    4. Bite-Sized Steps: Break complex fixes down. Each step must contain ONLY ONE primary action so the user does not get overwhelmed.
    5. Clear Expectations: For every step, explicitly define the 'Expected Result' so the user knows if the action was successful.
    6. Safety First: Never advise the user to delete critical system files, disable essential security protocols permanently, or perform hardware modifications while the device is plugged into power.

    Use the provided web snippets strictly to inform your troubleshooting plan, but format the output according to these rules."""
)

class Step(BaseModel):
    id: str
    title: str
    rationale: str
    action: str
    os: Optional[Literal["Windows", "macOS", "Linux", "Any"]] = "Any"
    commands: List[str] = Field(default_factory=list)
    expect: str = ""
    if_fails_next: Optional[str] = None

class TechMateOutput(BaseModel):
    issue_summary: str
    likely_causes: List[str] = Field(default_factory=list)
    plan_overview: List[str] = Field(default_factory=list)
    steps: List[Step] = Field(default_factory=list)
    confidence: float = 0.0

# --------------------------- 3) AI Web Search (Tavily) ---------------------------
def chunk_text_paragraphs(text: str, chunk_size: int = 1000) -> List[str]:
    if not text: return []
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip() and len(p.strip()) > 50]

async def fetch_tavily_context(query: str) -> List[str]:
    """Uses Tavily to search the web and extract clean text from the top results."""
    try:
        # 'advanced' depth performs deep scraping on the backend, 
        # include_raw_content=True returns the clean scraped text
        response = await tavily_client.search(
            query=query, 
            search_depth="advanced", 
            max_results=3, 
            include_raw_content=True
        )
        
        all_chunks = []
        for result in response.get("results", []):
            # Prefer raw_content if available, fallback to the summary snippet
            content = result.get("raw_content") or result.get("content")
            if content:
                all_chunks.extend(chunk_text_paragraphs(content))
        return all_chunks
    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return []

# --------------------------- 4) LLM Generation ---------------------------
async def ask_gemini_techmate(user_context: dict, page_snippets: List[dict]) -> TechMateOutput:
    model = genai.GenerativeModel(MODEL_NAME, system_instruction=TECHMATE_SYSTEM_PROMPT)
    prompt_text = (
        "Generate a full troubleshooting plan for the user's issue. Output ONLY valid JSON.\n\n"
        f"JSON Schema:\n{json.dumps(TechMateOutput.model_json_schema(), indent=2)}\n\n"
        f"User context:\n{json.dumps(user_context, indent=2)}\n\n"
        f"Web snippets:\n{json.dumps(page_snippets, indent=2)}"
    )
    
    try:
        resp = await model.generate_content_async(
            {"role": "user", "parts": [{"text": prompt_text}]},
            generation_config=genai.GenerationConfig(response_mime_type="application/json", temperature=0.2)
        )
        return TechMateOutput.model_validate_json(resp.text)
    except Exception as e:
        logger.error(f"Gemini generation failed: {e}")
        return TechMateOutput(issue_summary=user_context.get("query", "Error"), steps=[])

# --------------------------- 5) Main Agent Logic ---------------------------
# --------------------------- 5) Main Agent Logic ---------------------------
async def techmate_agent(query: str, device: str = "Windows laptop", os_name: str = "Windows") -> TechMateOutput:
    logger.info(f"Starting agent for query: {query}")
    
    # 1. Search Web & Get Clean Content (Tavily)
    all_chunks = await fetch_tavily_context(f"{os_name} {device} {query} troubleshooting fix")

    # 2. Add to Vector Store (Async)
    if all_chunks:
        await vector_store.add_texts_async(all_chunks)

    # 3. Retrieve & Re-Rank Relevant context
    # Grabs 10 initial chunks, but only gives the top 3 best to Gemini
    retrieved_chunks = await vector_store.search_and_rerank_async(
        query=query, 
        retrieve_top_k=10, 
        final_top_k=3
    )
    
    page_snippets = [{"url": "Tavily-RAG-source", "excerpt": chunk} for chunk in retrieved_chunks]

    # 4. Generate Plan
    user_ctx = {"query": query, "device": device, "os": os_name}
    plan = await ask_gemini_techmate(user_ctx, page_snippets)
    
    return plan