# backend/agent/chat_agent.py
import logging
import json
import os
import sqlite3
from typing import Dict, Any, TypedDict, Literal
from langgraph.graph import StateGraph, END
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# Import our core plan generator
from backend.agent.techmate_agent import techmate_agent

logger = logging.getLogger("techmate.chat_agent")

DB_PATH = "data/sessions.db"

# --- SQLite Session Management ---
def init_db():
    """Ensures the database and tables exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                state_data TEXT
            )
        ''')

def load_session(session_id: str) -> dict:
    """Loads a session state from SQLite."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute('SELECT state_data FROM sessions WHERE session_id = ?', (session_id,))
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return {"plan": {}, "current_step_index": 0}

def save_session(session_id: str, state_data: dict):
    """Saves or updates a session state in SQLite."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            INSERT INTO sessions (session_id, state_data)
            VALUES (?, ?)
            ON CONFLICT(session_id) DO UPDATE SET state_data = excluded.state_data
        ''', (session_id, json.dumps(state_data)))

def clear_session(session_id: str):
    """Deletes a session from SQLite (used when issue is resolved or exhausted)."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))

# --- Define the State for our Graph ---
class GraphState(TypedDict):
    session_id: str
    user_message: str
    device: str
    os_name: str
    plan: dict
    current_step_index: int
    intent: str
    reply: str

class TechMateChatAgent:
    def __init__(self):
        init_db()  # Initialize SQLite database on startup
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(GraphState)

        # Define Nodes
        workflow.add_node("classify_intent", self.node_classify_intent)
        workflow.add_node("generate_new_plan", self.node_generate_new_plan)
        workflow.add_node("execute_step", self.node_execute_step)
        workflow.add_node("handle_greeting", self.node_handle_greeting)
        workflow.add_node("resolve_issue", self.node_resolve_issue)

        # Define Routing Logic
        def route_intent(state: GraphState) -> str:
            intent = state.get("intent", "unknown")
            if intent == "greeting": return "handle_greeting"
            if intent == "affirmative": return "resolve_issue"
            if intent == "negative": return "execute_step"
            return "generate_new_plan"

        # Build Edges
        workflow.set_entry_point("classify_intent")
        workflow.add_conditional_edges(
            "classify_intent",
            route_intent,
            {
                "handle_greeting": "handle_greeting",
                "resolve_issue": "resolve_issue",
                "execute_step": "execute_step",
                "generate_new_plan": "generate_new_plan"
            }
        )
        
        workflow.add_edge("handle_greeting", END)
        workflow.add_edge("resolve_issue", END)
        workflow.add_edge("execute_step", END)
        workflow.add_edge("generate_new_plan", END)

        return workflow.compile()

    # --- Node Implementations ---
    
    def node_classify_intent(self, state: GraphState):
        msg = state["user_message"]
        has_plan = bool(state.get("plan"))
        
        system_prompt = f"""
        Classify the user's intent into exactly one of these categories: 
        'greeting', 'affirmative' (issue is fixed/worked), 'negative' (issue persists/didn't work), or 'new_issue' (describing a problem).
        Context: The user currently {"has" if has_plan else "does NOT have"} an active troubleshooting plan.
        Respond with ONLY the category word.
        """
        
        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=msg)
        ])
        
        return {"intent": response.content.strip().lower()}

    def node_handle_greeting(self, state: GraphState):
        return {"reply": "👋 Hi — I'm TechMate, your troubleshooting assistant. Tell me what's wrong with your device."}

    def node_resolve_issue(self, state: GraphState):
        # Clear DB session since resolved
        clear_session(state["session_id"])
        return {"reply": "✅ Great! It looks like your issue has been resolved. Let me know if you need help with anything else! 😊", "plan": {}, "current_step_index": 0}

    def node_execute_step(self, state: GraphState):
        plan_data = state.get("plan", {})
        steps = plan_data.get("steps", [])
        current_idx = state.get("current_step_index", 0)

        if current_idx + 1 < len(steps):
            next_idx = current_idx + 1
            next_step = steps[next_idx]
            reply = (
                f"**Let's try Step {next_idx + 1}: {next_step['title']}**\n\n"
                f"{next_step['action']}\n\n"
                f"💡 *Expected Result:* {next_step['expect']}"
            )
            return {"reply": reply, "current_step_index": next_idx}
        else:
            # Clear DB session since exhausted
            clear_session(state["session_id"])
            return {"reply": "⚠️ Sorry, we've exhausted the standard troubleshooting steps.\n\nI recommend visiting your nearest service center. You can start a new chat if you'd like to try diagnosing a different issue."}

    async def node_generate_new_plan(self, state: GraphState):
        logger.info(f"Generating new plan for session {state['session_id']}")
        
        plan = await techmate_agent(query=state["user_message"], device=state["device"], os_name=state["os_name"])
        plan_dict = plan.model_dump()
        first_step = plan_dict.get("steps", [])[0] if plan_dict.get("steps") else None
        
        if not first_step:
             reply = f"I couldn't find specific steps for: {plan.issue_summary}. Can you provide more details?"
        else:
             reply = (
                f"### 📋 Issue: {plan.issue_summary}\n\n"
                f"**Step 1: {first_step['title']}**\n"
                f"{first_step['action']}\n\n"
                f"💡 *Expected Result:* {first_step['expect']}\n\n"
                f"*(Reply 'done' if it worked, or 'it didn't work' to try the next step)*"
            )
        
        return {"reply": reply, "plan": plan_dict, "current_step_index": 0}

    # --- Main Entry Point ---
    async def handle_message(self, user_text: str, session_id: str, device: str, os_name: str) -> str:
        # Load session from SQLite Database instead of RAM
        session_data = load_session(session_id)
        
        initial_state: GraphState = {
            "session_id": session_id,
            "user_message": user_text,
            "device": device,
            "os_name": os_name,
            "plan": session_data.get("plan", {}),
            "current_step_index": session_data.get("current_step_index", 0),
            "intent": "",
            "reply": ""
        }

        # Run the graph
        final_state = await self.graph.ainvoke(initial_state)
        
        # Save updated state back to SQLite Database
        if final_state.get("plan"): 
            save_session(session_id, {
                "plan": final_state.get("plan"),
                "current_step_index": final_state.get("current_step_index")
            })

        return final_state["reply"]