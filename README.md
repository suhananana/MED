# 🏥 MediBot — Medical RAG Chatbot

A Streamlit web app powered by **Groq (Llama-3.3-70b)**, **LangGraph**, **FAISS**, and a live **web scraper** targeting Mayo Clinic, Healthline, and WebMD.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get a free Groq API key

Go to → https://console.groq.com  
Free tier: **100,000 tokens/day** — no credit card needed.

### 3. Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 🔧 How to use

1. Paste your Groq API key in the sidebar
2. Click **Build Knowledge Base** (scrapes ~17 medical pages, takes ~1–2 min)
3. Ask any medical question in the chat box

---

## 🛠️ Architecture

| Component | Technology |
|-----------|-----------|
| LLM | Groq — Llama-3.3-70b-versatile |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (local) |
| Vector Store | FAISS (in-memory) |
| Knowledge Base | Web scraper — Mayo Clinic · Healthline · WebMD |
| Orchestration | LangGraph ReAct Agent |
| Tools | RAG Retriever · Live Web Scraper · Symptom Checker |
| UI | Streamlit |

---

## ⚠️ Disclaimer

MediBot is for **educational and informational purposes only**.  
Always consult a qualified healthcare professional for medical advice.
