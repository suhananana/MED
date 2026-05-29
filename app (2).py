import streamlit as st
import os

# Load API key from .env (local) or Streamlit secrets (cloud)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if "ANTHROPIC_API_KEY" not in os.environ:
    try:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass

from tools.rag_router import RAGRouter
from tools.symptom_tool import SymptomCheckerTool
from tools.drug_tool import DrugInfoTool
from tools.anatomy_tool import AnatomyTool
from tools.emergency_tool import EmergencyTool
from tools.general_medical_tool import GeneralMedicalTool
import time

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediBot – AI Medical Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-primary: #0a0f1e;
    --bg-card: #0f1629;
    --bg-glass: rgba(15, 22, 41, 0.85);
    --accent-teal: #00d4aa;
    --accent-blue: #4facfe;
    --accent-red: #ff6b6b;
    --accent-amber: #ffd166;
    --text-primary: #e8f0fe;
    --text-muted: #7a8ba8;
    --border: rgba(79, 172, 254, 0.15);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080d1a 0%, #0a1020 100%);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent-teal);
}

/* Main chat area */
.main-header {
    text-align: center;
    padding: 2rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.main-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    background: linear-gradient(135deg, var(--accent-teal), var(--accent-blue));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.main-header p {
    color: var(--text-muted);
    font-size: 1rem;
    margin-top: 0.4rem;
    font-weight: 300;
}

/* Chat messages */
.user-message {
    background: linear-gradient(135deg, rgba(79,172,254,0.15), rgba(0,212,170,0.1));
    border: 1px solid rgba(79,172,254,0.25);
    border-radius: 18px 18px 4px 18px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    margin-left: 20%;
    position: relative;
}
.bot-message {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 18px 18px 18px 4px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    margin-right: 10%;
    position: relative;
}
.tool-badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 20px;
    margin-bottom: 0.6rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-symptom { background: rgba(255,107,107,0.2); color: #ff6b6b; border: 1px solid rgba(255,107,107,0.3); }
.badge-drug    { background: rgba(0,212,170,0.2);  color: #00d4aa; border: 1px solid rgba(0,212,170,0.3);  }
.badge-anatomy { background: rgba(79,172,254,0.2); color: #4facfe; border: 1px solid rgba(79,172,254,0.3); }
.badge-emergency { background: rgba(255,209,102,0.2); color: #ffd166; border: 1px solid rgba(255,209,102,0.3); }
.badge-general { background: rgba(161,135,255,0.2); color: #a187ff; border: 1px solid rgba(161,135,255,0.3); }

/* Sources section */
.sources-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    margin-top: 0.8rem;
    font-size: 0.82rem;
    color: var(--text-muted);
}
.sources-box strong { color: var(--accent-teal); }

/* Disclaimer */
.disclaimer {
    background: rgba(255,209,102,0.08);
    border: 1px solid rgba(255,209,102,0.2);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    font-size: 0.8rem;
    color: var(--accent-amber);
    margin-top: 0.5rem;
}

/* Input */
.stChatInputContainer textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
}
.stChatInputContainer textarea:focus {
    border-color: var(--accent-teal) !important;
    box-shadow: 0 0 0 2px rgba(0,212,170,0.15) !important;
}

/* Tool cards in sidebar */
.tool-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.6rem 0.9rem;
    margin: 0.4rem 0;
    font-size: 0.85rem;
}
.tool-card .tool-name { font-weight: 600; color: var(--accent-teal); }
.tool-card .tool-desc { color: var(--text-muted); font-size: 0.78rem; margin-top: 2px; }

/* Thinking indicator */
.thinking {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--text-muted);
    font-size: 0.85rem;
    padding: 0.5rem 0;
}
.dot {
    width: 8px; height: 8px;
    background: var(--accent-teal);
    border-radius: 50%;
    animation: pulse 1.4s ease-in-out infinite;
}
.dot:nth-child(2) { animation-delay: 0.2s; background: var(--accent-blue); }
.dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes pulse {
    0%, 80%, 100% { opacity: 0.3; transform: scale(0.8); }
    40% { opacity: 1; transform: scale(1); }
}
</style>
""", unsafe_allow_html=True)

# ─── Initialize Tools & Router ─────────────────────────────────────────────────
@st.cache_resource
def load_tools():
    tools = {
        "symptom": SymptomCheckerTool(),
        "drug": DrugInfoTool(),
        "anatomy": AnatomyTool(),
        "emergency": EmergencyTool(),
        "general": GeneralMedicalTool(),
    }
    router = RAGRouter(tools)
    return tools, router

tools, router = load_tools()

# ─── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tool_usage" not in st.session_state:
    st.session_state.tool_usage = {}

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 MediBot")
    st.markdown("**AI-Powered Medical Assistant**")
    st.markdown("---")

    st.markdown("### 🛠️ RAG Tools Active")
    tool_info = [
        ("🔴", "symptom", "Symptom Checker", "Analyzes symptoms & suggests conditions"),
        ("🟢", "drug",    "Drug Information", "Drug dosage, interactions & side effects"),
        ("🔵", "anatomy", "Anatomy Guide",    "Body systems & anatomical knowledge"),
        ("🟡", "emergency","Emergency Triage", "Urgent care & first-aid guidance"),
        ("🟣", "general", "General Medical",  "Broad medical Q&A knowledge base"),
    ]
    for emoji, key, name, desc in tool_info:
        count = st.session_state.tool_usage.get(key, 0)
        st.markdown(f"""
        <div class="tool-card">
            <div class="tool-name">{emoji} {name} {'— used ' + str(count) + 'x' if count else ''}</div>
            <div class="tool-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    show_sources = st.toggle("Show Sources", value=True)
    show_tool_used = st.toggle("Show Tool Used", value=True)
    temperature = st.slider("Response Detail", 0.0, 1.0, 0.3, 0.1,
                            help="Higher = more creative, Lower = more precise")

    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.tool_usage = {}
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:#7a8ba8; line-height:1.6">
    ⚠️ <strong>Disclaimer</strong><br>
    MediBot provides general medical information only. Always consult a qualified healthcare professional for diagnosis and treatment.
    </div>
    """, unsafe_allow_html=True)

# ─── Main Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🩺 MediBot</h1>
    <p>Intelligent Medical Assistant powered by multiple RAG knowledge tools</p>
</div>
""", unsafe_allow_html=True)

# ─── Chat History ──────────────────────────────────────────────────────────────
BADGE_MAP = {
    "symptom":   ("badge-symptom",   "🔴 Symptom Checker"),
    "drug":      ("badge-drug",      "🟢 Drug Info"),
    "anatomy":   ("badge-anatomy",   "🔵 Anatomy Guide"),
    "emergency": ("badge-emergency", "🟡 Emergency Triage"),
    "general":   ("badge-general",   "🟣 General Medical"),
}

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="user-message">
            <strong style="color:#4facfe">You</strong><br>{msg["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        tool_key = msg.get("tool", "general")
        badge_cls, badge_label = BADGE_MAP.get(tool_key, ("badge-general", "🟣 General"))
        tool_html = f'<span class="tool-badge {badge_cls}">{badge_label}</span><br>' if show_tool_used else ""
        sources_html = ""
        if show_sources and msg.get("sources"):
            srcs = "".join(f"<li>{s}</li>" for s in msg["sources"])
            sources_html = f'<div class="sources-box"><strong>📚 Sources:</strong><ul style="margin:4px 0 0 16px;padding:0">{srcs}</ul></div>'
        st.markdown(f"""
        <div class="bot-message">
            {tool_html}
            <strong style="color:#00d4aa">MediBot</strong><br>
            {msg["content"]}
            {sources_html}
        </div>
        """, unsafe_allow_html=True)

# ─── Suggestions ───────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("#### 💡 Try asking:")
    cols = st.columns(2)
    suggestions = [
        "What are symptoms of diabetes?",
        "What is ibuprofen used for?",
        "How does the heart work?",
        "What to do in case of a seizure?",
        "What causes high blood pressure?",
        "Is paracetamol safe during pregnancy?",
    ]
    for i, sug in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state.pending_query = sug
                st.rerun()

# ─── Chat Input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a medical question…")

# Handle suggestion click
if "pending_query" in st.session_state:
    user_input = st.session_state.pop("pending_query")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Show thinking
    thinking_placeholder = st.empty()
    thinking_placeholder.markdown("""
    <div class="thinking">
        <div class="dot"></div><div class="dot"></div><div class="dot"></div>
        <span>Routing to best RAG tool…</span>
    </div>
    """, unsafe_allow_html=True)

    # Route & get answer
    result = router.query(user_input, temperature=temperature)
    time.sleep(0.3)
    thinking_placeholder.empty()

    # Track tool usage
    tool_used = result.get("tool", "general")
    st.session_state.tool_usage[tool_used] = st.session_state.tool_usage.get(tool_used, 0) + 1

    # Save bot message
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "tool": tool_used,
        "sources": result.get("sources", []),
    })
    st.rerun()
