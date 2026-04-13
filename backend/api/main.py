# backend/api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging

from backend.agent.chat_agent import TechMateChatAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("techmate.api")

app = FastAPI(title="TechMate API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate our conversational agent
chat_agent = TechMateChatAgent()

class ChatRequest(BaseModel):
    session_id: str
    message: str
    device: str = "Windows laptop"
    os_name: str = "Windows"

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        # Pass the message to the conversational agent
        reply_text = await chat_agent.handle_message(
            user_text=req.message,
            session_id=req.session_id,
            device=req.device,
            os_name=req.os_name
        )
        
        # Return a simple text reply
        return {"status": "success", "reply": reply_text}
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")