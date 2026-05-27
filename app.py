import os
import re
import time
import requests
import streamlit as st
from bs4 import BeautifulSoup
from time import sleep

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="MediBot — Medical AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #f0f4f8; }
    .main-header {
        background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(26,115,232,0.3);
    }
    .chat-user {
        background: #e8f0fe;
        border-left: 4px solid #1a73e8;
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
    }
    .chat-bot {
        background: white;
        border-left: 4px solid #0d47a1;
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .tool-badge {
        display: inline-block;
        background: #e8f0fe;
        color: #1a73e8;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        margin: 2px;
        border: 1px solid #bbdefb;
    }
    .warning-box {
        background: #fff8e1;
        border: 1px solid #ffe082;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-size: 0.85rem;
        color: #795548;
        margin-top: 0.5rem;
    }
    div[data-testid="stChatInput"] { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h2 style="margin:0">🏥 MediBot — Medical AI Assistant</h2>
    <p style="margin:4px 0 0 0; opacity:0.85; font-size:0.95rem">
        Powered by Groq · LangGraph · FAISS · Web Scraper
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    groq_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get a free key at https://console.groq.com",
    )
    st.markdown("[Get free Groq key →](https://console.groq.com)", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🛠️ Tools Available")
    st.markdown("- 📚 **RAG Retriever** — FAISS local KB\n- 🌐 **Web Scraper** — Live Mayo/Healthline/WebMD\n- 🩺 **Symptom Checker** — Rule-based mapper")

    st.markdown("---")
    st.markdown("### 💡 Example Questions")
    examples = [
        "What is diabetes mellitus and how is it treated?",
        "I have fever, cough, and fatigue. What could it be?",
        "What are the side effects of ibuprofen?",
        "Explain hypertension symptoms and causes.",
        "How does chemotherapy work?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["prefill"] = ex

    st.markdown("---")
    build_kb = st.button("🔄 Build Knowledge Base", use_container_width=True, type="primary")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["history"] = []
        st.rerun()

# ── Session state ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "history" not in st.session_state:
    st.session_state["history"] = []
if "kb_ready" not in st.session_state:
    st.session_state["kb_ready"] = False
if "agent" not in st.session_state:
    st.session_state["agent"] = None

# ── Backend bootstrap ──────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

MEDICAL_URLS = [
    ("Diabetes mellitus",   "https://www.mayoclinic.org/diseases-conditions/diabetes/symptoms-causes/syc-20371444"),
    ("Hypertension",        "https://www.mayoclinic.org/diseases-conditions/high-blood-pressure/symptoms-causes/syc-20373410"),
    ("Asthma",              "https://www.mayoclinic.org/diseases-conditions/asthma/symptoms-causes/syc-20369653"),
    ("Heart disease",       "https://www.mayoclinic.org/diseases-conditions/heart-disease/symptoms-causes/syc-20353118"),
    ("Stroke",              "https://www.mayoclinic.org/diseases-conditions/stroke/symptoms-causes/syc-20350113"),
    ("Pneumonia",           "https://www.mayoclinic.org/diseases-conditions/pneumonia/symptoms-causes/syc-20354204"),
    ("Arthritis",           "https://www.mayoclinic.org/diseases-conditions/arthritis/symptoms-causes/syc-20350772"),
    ("Migraine",            "https://www.mayoclinic.org/diseases-conditions/migraine-headache/symptoms-causes/syc-20360201"),
    ("Alzheimer's disease", "https://www.mayoclinic.org/diseases-conditions/alzheimers-disease/symptoms-causes/syc-20350447"),
    ("Fever",               "https://www.healthline.com/health/fever"),
    ("Headache",            "https://www.healthline.com/health/headache"),
    ("Fatigue",             "https://www.healthline.com/health/fatigue"),
    ("Chest pain",          "https://www.healthline.com/health/chest-pain"),
    ("Cough",               "https://www.healthline.com/health/cough"),
    ("Ibuprofen",           "https://www.webmd.com/drugs/2/drug-5166/ibuprofen-oral/details"),
    ("Antibiotic",          "https://www.webmd.com/a-to-z-guides/what-are-antibiotics"),
    ("Mental health",       "https://www.healthline.com/health/mental-health"),
]


def scrape_page(title, url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
            tag.decompose()
        content = ""
        for sel in ["article", "main", '[class*="article"]', '[class*="content"]']:
            block = soup.select_one(sel)
            if block:
                content = block.get_text(separator=" ", strip=True)
                break
        if not content:
            content = soup.get_text(separator=" ", strip=True)
        content = re.sub(r"\s+", " ", content).strip()
        if len(content) < 100:
            return None
        from langchain_core.documents import Document
        return Document(page_content=content[:5000], metadata={"title": title, "source": url})
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def build_agent(api_key: str):
    """Build LangGraph agent — cached so it only runs once."""
    import os
    os.environ["GROQ_API_KEY"] = api_key

    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.tools import create_retriever_tool, tool
    from langchain_core.documents import Document
    from langchain_core.messages import SystemMessage, AIMessage
    from typing import List
    from typing_extensions import TypedDict, Annotated
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode

    # --- scrape KB ---
    docs = []
    for title, url in MEDICAL_URLS:
        doc = scrape_page(title, url)
        if doc:
            docs.append(doc)
        time.sleep(0.3)

    if not docs:
        raise RuntimeError("No documents scraped. Check internet connection.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    rag_tool = create_retriever_tool(
        retriever,
        name="medical_rag_retriever",
        description=(
            "Search the local medical knowledge base scraped from Mayo Clinic, "
            "Healthline, and WebMD. Use this FIRST for any medical question."
        ),
    )

    @tool
    def web_scraper_tool(query: str) -> str:
        """Scrape live medical info from Mayo Clinic and Healthline for a given query."""
        import urllib.parse
        sources = [
            ("Mayo Clinic", f"https://www.mayoclinic.org/search/search-results?q={urllib.parse.quote(query)}"),
            ("Healthline",  f"https://www.healthline.com/search?q1={urllib.parse.quote(query)}"),
        ]
        results = []
        for source_name, search_url in sources:
            try:
                resp = requests.get(search_url, headers=HEADERS, timeout=8)
                if resp.status_code != 200:
                    continue
                soup = BeautifulSoup(resp.text, "lxml")
                link = None
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    if source_name == "Mayo Clinic" and "/diseases-conditions/" in href:
                        link = href if href.startswith("http") else "https://www.mayoclinic.org" + href
                        break
                    elif source_name == "Healthline" and "/health/" in href and href.startswith("/health/"):
                        link = "https://www.healthline.com" + href
                        break
                if not link:
                    continue
                page_resp = requests.get(link, headers=HEADERS, timeout=8)
                page_soup = BeautifulSoup(page_resp.text, "lxml")
                for tag in page_soup(["script", "style", "nav", "footer", "header", "aside"]):
                    tag.decompose()
                content = ""
                for sel in ["article", "main", '[class*="article"]', '[class*="content"]']:
                    block = page_soup.select_one(sel)
                    if block:
                        content = block.get_text(separator=" ", strip=True)
                        break
                if not content:
                    content = page_soup.get_text(separator=" ", strip=True)
                content = re.sub(r"\s+", " ", content).strip()
                if len(content) > 200:
                    results.append(f"[{source_name} — {link}]\n\n{content[:2000]}")
                    break
            except Exception:
                continue
        return "\n\n---\n\n".join(results) if results else f"No results for '{query}'."

    @tool
    def symptom_checker(symptoms: str) -> str:
        """Given a comma-separated list of symptoms, return possible medical conditions."""
        symptom_map = {
            "fever":               ["Influenza", "COVID-19", "Malaria", "Typhoid"],
            "cough":               ["Asthma", "COVID-19", "Tuberculosis", "Bronchitis"],
            "chest pain":          ["Heart disease", "Angina", "Pneumonia", "GERD"],
            "headache":            ["Migraine", "Hypertension", "Tension headache"],
            "fatigue":             ["Anemia", "Diabetes", "Hypothyroidism", "Depression"],
            "shortness of breath": ["Asthma", "Heart failure", "COVID-19"],
            "weight loss":         ["Diabetes", "Cancer", "Tuberculosis"],
            "joint pain":          ["Arthritis", "Gout", "Lupus", "Fibromyalgia"],
            "nausea":              ["Gastritis", "Food poisoning", "Migraine"],
            "dizziness":           ["Hypertension", "Anemia", "Inner ear disorder"],
            "abdominal pain":      ["Gastritis", "Appendicitis", "IBS", "Kidney stones"],
        }
        entered = [s.strip().lower() for s in symptoms.split(",")]
        result = {}
        for sym in entered:
            for key, conds in symptom_map.items():
                if key in sym:
                    result[key] = conds
        if not result:
            return "No matches found. Please describe symptoms clearly or consult a healthcare professional."
        lines = ["Possible conditions (educational only):\n"]
        for sym, conds in result.items():
            lines.append(f"  • {sym.capitalize()}: {', '.join(conds)}")
        lines.append("\nAlways consult a licensed physician for proper diagnosis.")
        return "\n".join(lines)

    tools_list = [rag_tool, web_scraper_tool, symptom_checker]

    SYSTEM_PROMPT = """You are MediBot, an expert Medical AI Assistant.

Answer medical questions clearly, accurately, and in detail.

Tools available:
1. medical_rag_retriever — search local FAISS knowledge base
2. web_scraper_tool — scrape live from Mayo Clinic / Healthline / WebMD
3. symptom_checker — map symptoms to conditions

Rules:
- ALWAYS use medical_rag_retriever first.
- Use symptom_checker if symptoms are mentioned.
- Use web_scraper_tool for additional live details.
- Give structured and detailed answers.

Always end with:
⚠️ For informational purposes only. Always consult a healthcare professional."""

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0,
        max_tokens=1024,
    )
    llm_with_tools = llm.bind_tools(tools_list)
    tool_node = ToolNode(tools_list)

    class AgentState(TypedDict):
        messages:   Annotated[list, add_messages]
        tools_used: List[str]

    def call_model(state: AgentState):
        msgs = list(state["messages"])
        if not any(isinstance(m, SystemMessage) for m in msgs):
            msgs = [SystemMessage(content=SYSTEM_PROMPT)] + msgs
        response = llm_with_tools.invoke(msgs)
        return {"messages": [response]}

    def run_tools(state: AgentState):
        last = state["messages"][-1]
        used = list(state.get("tools_used", []))
        if hasattr(last, "tool_calls") and last.tool_calls:
            for tc in last.tool_calls:
                name = tc.get("name", "unknown")
                if name not in used:
                    used.append(name)
        result = tool_node.invoke(state)
        result["tools_used"] = used
        return result

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", run_tools)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(), len(docs), vectorstore.index.ntotal


TOOL_LABELS = {
    "medical_rag_retriever": "📚 RAG Retriever",
    "web_scraper_tool":      "🌐 Web Scraper",
    "symptom_checker":       "🩺 Symptom Checker",
}

# ── Build KB button ────────────────────────────────────────────
if build_kb:
    if not groq_key:
        st.sidebar.error("Please enter your Groq API key first.")
    else:
        with st.spinner("🔄 Scraping knowledge base & building FAISS index… (takes ~1–2 min)"):
            try:
                agent, n_docs, n_vecs = build_agent(groq_key)
                st.session_state["agent"] = agent
                st.session_state["kb_ready"] = True
                st.sidebar.success(f"✅ KB ready! {n_docs} articles · {n_vecs} vectors")
            except Exception as e:
                st.sidebar.error(f"❌ {e}")

# ── Display conversation ───────────────────────────────────────
chat_container = st.container()
with chat_container:
    if not st.session_state["messages"]:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 1rem; color: #555;">
            <h3>👋 Welcome to MediBot!</h3>
            <p>Enter your Groq API key in the sidebar, click <b>Build Knowledge Base</b>, then ask away.</p>
            <p style="font-size:0.9rem; color:#888;">
                Covers diseases, symptoms, medications, treatments & more<br>
                sourced from Mayo Clinic · Healthline · WebMD
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">🧑 <b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                tools_html = ""
                if msg.get("tools"):
                    badges = "".join(
                        f'<span class="tool-badge">{TOOL_LABELS.get(t, t)}</span>'
                        for t in msg["tools"]
                    )
                    tools_html = f'<div style="margin-top:8px">🛠️ {badges}</div>'
                st.markdown(
                    f'<div class="chat-bot">🤖 <b>MediBot:</b><br>{msg["content"]}{tools_html}</div>',
                    unsafe_allow_html=True,
                )

# ── Chat input ─────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", "")
question = st.chat_input("Ask a medical question…") or prefill

if question:
    if not groq_key:
        st.error("Please enter your Groq API key in the sidebar.")
    elif not st.session_state["kb_ready"]:
        st.error("Please click **Build Knowledge Base** in the sidebar first.")
    else:
        st.session_state["messages"].append({"role": "user", "content": question})

        from langchain_core.messages import HumanMessage, AIMessage
        st.session_state["history"].append(HumanMessage(content=question))

        with st.spinner("MediBot is thinking…"):
            try:
                agent = st.session_state["agent"]
                result = agent.invoke(
                    {"messages": st.session_state["history"], "tools_used": []},
                    config={"recursion_limit": 20},
                )
                ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
                if ai_msgs:
                    final = ai_msgs[-1]
                    reply = final.content if isinstance(final.content, str) else str(final.content)
                    tools_used = result.get("tools_used", [])
                    st.session_state["history"].append(final)
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": reply,
                        "tools": tools_used,
                    })
                else:
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": "No response received. Please try again.",
                        "tools": [],
                    })
            except Exception as e:
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": f"❌ Error: {e}",
                    "tools": [],
                })
        st.rerun()

# ── Footer ─────────────────────────────────────────────────────
st.markdown("""
<div class="warning-box">
    ⚠️ <b>Disclaimer:</b> MediBot is for informational and educational purposes only.
    It is not a substitute for professional medical advice, diagnosis, or treatment.
    Always consult a qualified healthcare provider.
</div>
""", unsafe_allow_html=True)
