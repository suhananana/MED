# 🩺 MediBot — AI Medical Chatbot with RAG

A Streamlit-based medical chatbot powered by multiple RAG (Retrieval-Augmented Generation) tools and Claude AI.

## 🛠️ RAG Tools

| Tool | Trigger | Knowledge Base |
|------|---------|----------------|
| 🔴 **Symptom Checker** | Symptoms, pain, fever, rash | 10 symptom entries |
| 🟢 **Drug Information** | Drug names, dosage, side effects | 10 drug profiles |
| 🔵 **Anatomy Guide** | Organs, body systems, physiology | 10 anatomy entries |
| 🟡 **Emergency Triage** | Emergency, urgent, first aid | 10 emergency protocols |
| 🟣 **General Medical** | Everything else | 10 general health entries |

The **RAG Router** automatically detects which tool to use based on keyword scoring — no extra API calls needed for routing.

## 🚀 Quick Start

### 1. Clone / Download the project

```bash
cd medbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your API key

**Option A — .env file (local dev):**
```bash
cp .env.example .env
# Edit .env and add your key:
# ANTHROPIC_API_KEY=sk-ant-...
```

**Option B — Streamlit secrets (Streamlit Cloud):**
Edit `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

### 4. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## ☁️ Deploy to Streamlit Cloud

1. Push this folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → set `app.py` as the main file
4. Add `ANTHROPIC_API_KEY` in **Secrets** (Settings → Secrets)
5. Click **Deploy**

## 📁 Project Structure

```
medbot/
├── app.py                    # Main Streamlit UI
├── requirements.txt
├── .env.example
├── .streamlit/
│   ├── config.toml           # Theme & server config
│   └── secrets.toml          # API keys (don't commit!)
└── tools/
    ├── __init__.py
    ├── base_tool.py          # Abstract RAG base class
    ├── rag_router.py         # Query → tool routing logic
    ├── symptom_tool.py       # 🔴 Symptom Checker
    ├── drug_tool.py          # 🟢 Drug Information
    ├── anatomy_tool.py       # 🔵 Anatomy Guide
    ├── emergency_tool.py     # 🟡 Emergency Triage
    └── general_medical_tool.py # 🟣 General Medical
```

## 🔧 Extending the Knowledge Base

Each tool has a `_load_knowledge_base()` method returning a list of dicts:

```python
def _load_knowledge_base(self):
    return [
        {
            "text": "Your medical content here...",
            "source": "Source name for citation",
        },
        # Add more entries...
    ]
```

For production, replace the in-memory list with:
- **FAISS** or **ChromaDB** for vector similarity search
- **LangChain** or **LlamaIndex** for advanced RAG pipelines
- Real medical databases (PubMed, DrugBank, etc.)

## ⚠️ Disclaimer

MediBot is for educational and informational purposes only. It does not provide medical diagnosis or replace professional medical advice. Always consult a qualified healthcare professional.
