# TechMate - AI-Powered Research Assistant

TechMate is a Streamlit-based AI-powered research assistant that combines **web search** and **Retrieval-Augmented Generation (RAG)** to provide intelligent, structured, and summarized answers.  
It helps users query information, get summarized results, and perform advanced research tasks seamlessly.

---

## 🚀 Features

- 🔎 **Web Search Integration** – Search live information from the web.  
- 📄 **RAG (Retrieval-Augmented Generation)** – Fetch, embed, and query documents.  
- 🧾 **Structured Output** – Responses in JSON format (title, URL, snippet).  
- 📝 **Summarization** – Concise answers for user queries.  
- 💬 **Interactive UI** – Simple, intuitive Streamlit interface.  

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** – UI framework  
- **FAISS** – Vector database for embeddings  
- **SentenceTransformers** – For text embeddings  
- **Google Generative AI (Gemini API)** – LLM for reasoning and summarization  
- **SerpAPI / Web Scraping (httpx + BeautifulSoup)** – Search results  


## 🏗️ System Design

1. **User Interaction:** Users send tech-related queries through the interface 💬.
2. **Query Processing:** Queries are sent to the Node.js server, which passes them to LLM Gemini with a system prompt for structured answers 🧠.
3. **Data Retrieval:** Gemini fetches and organizes relevant web search results 🌐.
4. **Stepwise Response:** Solutions are presented step by step, automatically progressing if one step fails 🔄.
5. **Response Delivery:** Curated answers are sent back to the user in a clear, conversational format 📩.

---


## 🙋‍♀️ Author

<table>
  <tr>
    <td>
      <strong>Ishita Amin</strong><br/>
      👩‍💻 B.Tech CSE @ Navrachana University<br/>
      📬 <a href="mailto:aminishita30@gmail.com">aminishita30@gmail.com</a><br/>
  🔗 <a href="https://www.linkedin.com/in/ishita-amin-841726253" target="_blank">LinkedIn</a><br/>    </td>
  </tr>
</table>

---

