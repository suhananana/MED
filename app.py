import os
import re
import time
import requests
import streamlit as st
from bs4 import BeautifulSoup

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Global dark background */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: #0e0e0e !important;
        color: #e0e0e0 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #141414 !important;
        border-right: 1px solid #2a2a2a;
    }
    [data-testid="stSidebar"] * { color: #cccccc !important; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1565c0 0%, #0a2a6e 100%);
        padding: 1.4rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(21,101,192,0.4);
    }

    /* Chat bubbles */
    .chat-user {
        background: #1a1a2e;
        border-left: 3px solid #1565c0;
        border-radius: 0 10px 10px 0;
        padding: 0.8rem 1.1rem;
        margin: 0.6rem 0;
        color: #d0d8ff;
    }
    .chat-bot {
        background: #161616;
        border-left: 3px solid #0a4a9e;
        border-radius: 0 10px 10px 0;
        padding: 0.8rem 1.1rem;
        margin: 0.6rem 0;
        color: #e0e0e0;
        white-space: pre-wrap;
    }

    /* Sidebar inputs */
    .stTextInput input {
        background-color: #1e1e1e !important;
        color: #e0e0e0 !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
    }
    .stButton button {
        background-color: #1565c0 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
    }
    .stButton button:hover { background-color: #1976d2 !important; }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        background-color: #1a1a1a !important;
        color: #e0e0e0 !important;
        border: 1px solid #333 !important;
        border-radius: 12px !important;
    }
    [data-testid="stChatInput"] {
        background-color: #1a1a1a !important;
    }

    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #111; }
    ::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h2 style="margin:0">🏥 Medical AI Assistant</h2>
    <p style="margin:4px 0 0 0; opacity:0.8; font-size:0.9rem">
        Ask any health or medical question
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

# ── Scraping helpers ───────────────────────────────────────────
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
    import os
    os.environ["GROQ_API_KEY"] = api_key

    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.tools import create_retriever_tool, tool
    from langchain_core.messages import SystemMessage, AIMessage
    from typing import List
    from typing_extensions import TypedDict, Annotated
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode

    docs = []
    for title, url in MEDICAL_URLS:
        doc = scrape_page(title, url)
        if doc:
            docs.append(doc)
        time.sleep(0.3)

    # Fallback minimal docs if scraping fails
    if not docs:
        from langchain_core.documents import Document
        docs = [Document(
            page_content="General medical information. Common diseases include diabetes, hypertension, asthma, heart disease, stroke, pneumonia, arthritis, migraine, and Alzheimer's. Common symptoms include fever, headache, fatigue, chest pain, and cough.",
            metadata={"title": "General Medicine", "source": "internal"}
        )]

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
            "Search the local medical knowledge base from Mayo Clinic, Healthline, WebMD. "
            "Use this FIRST for any medical question."
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
                    results.append(f"[{source_name}]\n\n{content[:2000]}")
                    break
            except Exception:
                continue
        return "\n\n---\n\n".join(results) if results else ""

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
            return ""
        lines = []
        for sym, conds in result.items():
            lines.append(f"{sym.capitalize()}: {', '.join(conds)}")
        return "\n".join(lines)

    tools_list = [rag_tool, web_scraper_tool, symptom_checker]

    SYSTEM_PROMPT = """You are a knowledgeable Medical AI Assistant. Your job is to always provide helpful, clear, and detailed general medical information.

Tools available:
1. medical_rag_retriever — search local medical knowledge base (use this first)
2. web_scraper_tool — fetch live info from trusted medical sites
3. symptom_checker — map symptoms to possible conditions

STRICT RULES:
- ALWAYS give a real, informative answer about the medical topic asked.
- If a question mentions a person (e.g. "Tim Cook's health"), IGNORE the person entirely and answer the general medical topic instead. For example if asked about someone's diabetes, explain diabetes in general.
- NEVER say you need to search for someone's personal health info.
- NEVER mention what tools you are going to use or have used.
- NEVER say you "would need to search" or "cannot find" anything.
- NEVER refer to yourself as MediBot or any specific name — just answer directly.
- Always answer as if explaining to a patient — clear, warm, educational.
- Use your tools silently in the background; the user should only see the final medical answer.
- Format with sections/bullet points when the answer benefits from structure.
- End with one line: "Note: This is general health information. Consult a doctor for personal advice."
"""

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


# ── Build KB button ────────────────────────────────────────────
if build_kb:
    if not groq_key:
        st.sidebar.error("Please enter your Groq API key first.")
    else:
        with st.spinner("Building knowledge base… this takes about a minute."):
            try:
                agent, n_docs, n_vecs = build_agent(groq_key)
                st.session_state["agent"] = agent
                st.session_state["kb_ready"] = True
                st.sidebar.success(f"✅ Ready! {n_docs} articles loaded.")
            except Exception as e:
                st.sidebar.error(f"❌ {e}")

# ── Display conversation ───────────────────────────────────────
with st.container():
    if not st.session_state["messages"]:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 1rem; color: #666;">
            <h3 style="color:#aaa;">Ask me anything about health & medicine</h3>
            <p style="color:#555;">Enter your Groq API key in the sidebar and click <b style="color:#aaa;">Build Knowledge Base</b> to get started.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">🧑&nbsp; {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                content = msg["content"].replace("\n", "<br>")
                st.markdown(
                    f'<div class="chat-bot">🤖&nbsp; {content}</div>',
                    unsafe_allow_html=True,
                )

# ── Chat input ─────────────────────────────────────────────────
question = st.chat_input("Ask a medical question…")

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
                    st.session_state["history"].append(final)
                else:
                    reply = "I'm here to help with your medical question. Could you please provide more details so I can give you the best information?"
                st.session_state["messages"].append({"role": "assistant", "content": reply})
            except Exception:
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": "I can help with general medical information. Please try rephrasing your question.",
                })
        st.rerun()
