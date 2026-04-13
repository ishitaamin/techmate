# frontend/app.py
import streamlit as st
import requests
import uuid
import os

# Configuration
API_URL = os.getenv("API_URL", "http://backend:8000/api/chat")

# 1. Page Configuration
st.set_page_config(
    page_title="TechMate AI", 
    page_icon="⚡", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS for a "SaaS" Look
st.markdown("""
    <style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Safely hide Streamlit junk without breaking the sidebar toggle */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stHeader"] {background-color: transparent;}
    
    /* Gradient Title */
    .gradient-text {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0px;
    }
    
    /* Subtitle styling */
    .sub-text {
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Floating chat input */
    .stChatInputContainer {
        border-radius: 20px !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05) !important;
        border: 1px solid #e5e7eb !important;
        padding: 5px !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
        background-color: #f9fafb;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Custom Header
st.markdown('<h1 class="gradient-text">⚡ TechMate.ai</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Your intelligent, agentic IT support assistant.</p>', unsafe_allow_html=True)

# 4. Sidebar for Context Management
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8649/8649595.png", width=60) # Cool tech icon
    st.header("⚙️ Diagnostics Context")
    st.markdown("Help TechMate tailor its solutions to your specific environment.")
    
    with st.container(border=True):
        device_type = st.selectbox("🖥️ Device", ["Windows laptop", "MacBook", "iPhone", "Android Phone", "Desktop PC"])
        os_name = st.selectbox("💿 OS", ["Windows", "macOS", "iOS", "Android", "Linux"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🗑️ Clear Conversation", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
        
    st.caption("Session ID: " + st.session_state.get("session_id", "N/A")[:8])

# 5. Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# 6. Default Welcome Message (Empty State)
if len(st.session_state.messages) == 0:
    with st.chat_message("assistant", avatar="⚡"):
        st.markdown(
            f"**Welcome to TechMate!** 👋 \n\n"
            f"I see you're using a **{device_type}** running **{os_name}**. \n\n"
            "I'm equipped with live web-search and diagnostic reasoning. Describe your issue below, for example:\n"
            "- *\"My screen is flickering when I open Chrome\"*\n"
            "- *\"My WiFi keeps dropping randomly\"*\n"
            "- *\"My device is overheating during normal use\"*"
        )

# 7. Render Chat History with Custom Avatars
for msg in st.session_state.messages:
    # User gets a human avatar, Assistant gets the lightning bolt
    avatar_icon = "🧑‍💻" if msg["role"] == "user" else "⚡"
    with st.chat_message(msg["role"], avatar=avatar_icon):
        st.markdown(msg["content"])

# 8. Chat Input & API Communication
if prompt := st.chat_input("Describe your tech issue here..."):
    
    # Render user message instantly
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # Call Backend API
    with st.chat_message("assistant", avatar="⚡"):
        with st.spinner("Analyzing issue and scanning knowledge bases..."):
            try:
                payload = {
                    "session_id": st.session_state.session_id,
                    "message": prompt,
                    "device": device_type,
                    "os_name": os_name
                }
                
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    reply = response.json().get("reply", "I encountered an error formatting the response.")
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                else:
                    st.error(f"⚠️ Backend API Error (Status {response.status_code})")
                    
            except requests.exceptions.ConnectionError:
                st.error(f"🔌 Failed to connect to the backend at `{API_URL}`. Make sure your Docker container or FastAPI server is running.")