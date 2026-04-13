# ⚡ TechMate AI: Autonomous IT Support Agent

TechMate is an autonomous, agentic RAG (Retrieval-Augmented Generation) troubleshooting assistant. Rather than relying on simple scripts, it utilizes a state-machine workflow to diagnose, search, and guide users through physical hardware and software fixes.

## 🏗️ Architecture & Tech Stack

* **Agentic Routing:** LangGraph (State machine for intent classification and progressive troubleshooting).
* **LLM Engine:** Google Gemini 2.5 Flash.
* **Search Engine:** Tavily API (Optimized web extraction for AI agents).
* **Vector Database:** FAISS with SentenceTransformers (Retrieve & Re-rank architecture via Cross-Encoders).
* **Backend:** FastAPI (Fully asynchronous to prevent blocking during heavy ML operations).
* **Frontend:** Streamlit.
* **State Management:** SQLite (Persistent user sessions across server restarts).
* **Deployment:** Docker & Docker Compose (Multi-container orchestration).

## ✨ Key Features
1. **Intelligent Intent Routing:** Understands if a user is succeeding, failing, or starting a new issue, and dynamically alters the troubleshooting plan.
2. **Re-Ranked RAG:** Uses a Bi-encoder to quickly cast a wide net, and a Cross-encoder to score and filter the absolute best context to inject into the LLM.
3. **Action-Oriented Directives:** Heavily constrained system prompts ensure the agent provides concrete, physical steps rather than open-ended diagnostic questions.
4. **Persistent Sessions:** User conversation state is safely stored in an internal SQLite database, protecting active troubleshooting sessions from container restarts.

## 🚀 Local Quickstart

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/techmate.git](https://github.com/YOUR_USERNAME/techmate.git)
   cd techmate
   ```
2. **Set up your environment variables:**
    ```bash
    GEMINI_API_KEY="your_google_api_key"
    TAVILY_API_KEY="your_tavily_api_key"
    ```
3. **Spin up the containers:**
    ```bash
    docker-compose up --build -d
    ```
4. **Access the App:**
    Open http://localhost:8501 in your browser.

live 

## 🙋‍♀️ Author

<table>
  <tr>
    <td>
      <strong>Ishita Amin</strong><br/>
      👩‍💻 B.Tech CSE @ Navrachana University<br/>
      📬 <a href="mailto:aminishita30@gmail.com">aminishita30@gmail.com</a><br/>
      🔗 <a href="[https://linkedin.com/in/ishitaamin](https://www.linkedin.com/in/ishita-amin-841726253)" target="_blank">LinkedIn</a><br/>
    </td>
  </tr>
</table>

---