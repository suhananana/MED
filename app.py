import os
import streamlit as st
from time import sleep
from collections import defaultdict
from typing import Annotated, List
from typing_extensions import TypedDict

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="🏥 MediBot — Medical RAG Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports (heavy; cached so they only run once) ───────────────────────
@st.cache_resource(show_spinner="🔧 Loading AI models & knowledge base (first run ~2 min)…")
def initialise(groq_api_key: str):
    """Build the full RAG pipeline and return the agent + helpers."""

    os.environ["GROQ_API_KEY"] = groq_api_key

    import wikipediaapi
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.tools import tool
    from langchain_core.documents import Document
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.retrievers import BaseRetriever
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode
    from pydantic import ConfigDict

    # ── Wikipedia knowledge base ─────────────────────────────────────────────
    MEDICAL_TOPICS = [
        'Diabetes mellitus', 'Hypertension', 'COVID-19', 'Cancer', 'Asthma',
        'Tuberculosis', 'Pneumonia', 'Influenza', 'Dengue fever', 'Malaria',
        'Heart disease', 'Stroke', 'Kidney failure', 'Liver disease', 'Obesity',
        'Anemia', 'Arthritis', 'Migraine', 'Epilepsy',
        "Parkinson's disease", "Alzheimer's disease",
        'Human body', 'Brain', 'Heart', 'Lung', 'Kidney', 'Liver',
        'Digestive system', 'Nervous system', 'Immune system',
        'Respiratory system', 'Circulatory system', 'Endocrine system',
        'Fever', 'Cough', 'Chest pain', 'Headache', 'Fatigue',
        'Shortness of breath', 'Abdominal pain', 'Diarrhea', 'Weight loss',
        'Antibiotic', 'Vaccination', 'Insulin', 'Chemotherapy',
        'Paracetamol', 'Ibuprofen', 'Pain management', 'Metformin',
        'Aspirin', 'Amoxicillin',
        'Nutrition', 'Vitamin', 'Mental health', 'Exercise', 'Public health',
    ]

    wiki = wikipediaapi.Wikipedia(language='en', user_agent='MediBot/2.0')

    def fetch_page(title):
        try:
            page = wiki.page(title)
            if page.exists():
                return Document(page_content=page.text[:5000],
                                metadata={'title': page.title, 'source': page.fullurl})
        except Exception:
            pass
        return None

    all_docs = []
    progress = st.progress(0, text="📚 Loading Wikipedia medical knowledge base…")
    for idx, topic in enumerate(MEDICAL_TOPICS):
        doc = fetch_page(topic)
        if doc:
            all_docs.append(doc)
        progress.progress((idx + 1) / len(MEDICAL_TOPICS),
                          text=f"📚 Loading: {topic} ({idx+1}/{len(MEDICAL_TOPICS)})")
        sleep(0.3)
    progress.empty()

    # ── Hybrid RAG ───────────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={'k': 4})

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4

    # ── RRF ensemble ─────────────────────────────────────────────────────────
    def reciprocal_rank_fusion(results, k=60):
        fused_scores = defaultdict(float)
        unique_docs = {}
        for result_list in results:
            for rank, doc in enumerate(result_list):
                doc_id = (doc.page_content, doc.metadata.get('source'))
                unique_docs.setdefault(doc_id, doc)
                fused_scores[doc_id] += 1 / (k + rank)
        return [unique_docs[d] for d in sorted(fused_scores, key=fused_scores.get, reverse=True)]

    class CustomEnsembleRetriever(BaseRetriever):
        retrievers: list
        weights: list = None
        k: int = 60
        model_config = ConfigDict(arbitrary_types_allowed=True)

        def _get_relevant_documents(self, query):
            return reciprocal_rank_fusion([r.invoke(query) for r in self.retrievers], self.k)

        async def _aget_relevant_documents(self, query):
            raise NotImplementedError

    ensemble_retriever = CustomEnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever], weights=[0.6, 0.4])

    # ── Tools ────────────────────────────────────────────────────────────────
    @tool
    def faiss_rag_retriever(query: str) -> str:
        """Semantic/dense vector search in local FAISS medical knowledge base."""
        return faiss_retriever.invoke(query)

    @tool
    def bm25_rag_retriever(query: str) -> str:
        """Keyword BM25 search in local medical knowledge base."""
        return bm25_retriever.invoke(query)

    @tool
    def hybrid_rag_retriever(query: str) -> str:
        """Hybrid RAG: FAISS semantic + BM25 keyword via Reciprocal Rank Fusion. PRIMARY retriever."""
        return ensemble_retriever.invoke(query)

    @tool
    def wikipedia_live_tool(query: str) -> str:
        """Search Wikipedia live for medical information. Use as fallback."""
        wc = wikipediaapi.Wikipedia(language='en', user_agent='MediBot/2.0')
        try:
            page = wc.page(query)
            if page.exists():
                return f"[Source: {page.fullurl}]\n\n{page.text[:3000]}"
            return f"No Wikipedia page found for '{query}'."
        except Exception as e:
            return f'Wikipedia error: {e}'

    @tool
    def symptom_checker(symptoms: str) -> str:
        """Analyze comma-separated symptoms; return possible conditions with urgency levels."""
        symptom_map = {
            'fever':               {'conditions': ['Influenza', 'COVID-19', 'Malaria', 'Typhoid', 'Dengue'], 'urgency': 'Medium'},
            'cough':               {'conditions': ['Asthma', 'COVID-19', 'Tuberculosis', 'Bronchitis'], 'urgency': 'Low-Medium'},
            'chest pain':          {'conditions': ['Heart disease', 'Angina', 'Pneumonia', 'GERD'], 'urgency': '🚨 HIGH — Seek immediate care'},
            'headache':            {'conditions': ['Migraine', 'Hypertension', 'Tension headache', 'Meningitis'], 'urgency': 'Low-Medium'},
            'fatigue':             {'conditions': ['Anemia', 'Diabetes', 'Hypothyroidism', 'Depression'], 'urgency': 'Low'},
            'shortness of breath': {'conditions': ['Asthma', 'Heart failure', 'COVID-19', 'Pulmonary embolism'], 'urgency': '🚨 HIGH — Seek immediate care'},
            'frequent urination':  {'conditions': ['Diabetes mellitus', 'UTI', 'Prostate issues'], 'urgency': 'Medium'},
            'weight loss':         {'conditions': ['Diabetes', 'Cancer', 'Tuberculosis', 'Hyperthyroidism'], 'urgency': 'Medium-High'},
            'joint pain':          {'conditions': ['Arthritis', 'Gout', 'Lupus', 'Fibromyalgia'], 'urgency': 'Low-Medium'},
            'rash':                {'conditions': ['Eczema', 'Psoriasis', 'Allergic reaction', 'Lupus'], 'urgency': 'Low'},
            'nausea':              {'conditions': ['Gastritis', 'Food poisoning', 'Pregnancy', 'Migraine'], 'urgency': 'Low-Medium'},
            'dizziness':           {'conditions': ['Hypertension', 'Anemia', 'Inner ear disorder', 'Stroke'], 'urgency': 'Medium'},
            'vomiting':            {'conditions': ['Gastritis', 'Food poisoning', 'Appendicitis', 'Migraine'], 'urgency': 'Medium'},
            'abdominal pain':      {'conditions': ['Gastritis', 'Appendicitis', 'IBS', 'Kidney stones'], 'urgency': 'Medium-High'},
            'back pain':           {'conditions': ['Muscle strain', 'Herniated disc', 'Kidney stones', 'Sciatica'], 'urgency': 'Low-Medium'},
            'swelling':            {'conditions': ['Heart failure', 'Kidney disease', 'DVT', 'Lymphedema'], 'urgency': 'Medium'},
            'blurred vision':      {'conditions': ['Diabetes', 'Hypertension', 'Glaucoma', 'Stroke'], 'urgency': '🚨 HIGH — Seek immediate care'},
            'confusion':           {'conditions': ['Stroke', 'Hypoglycemia', 'Dementia', 'Encephalitis'], 'urgency': '🚨 HIGH — Seek immediate care'},
        }
        entered = [s.strip().lower() for s in symptoms.split(',')]
        result = {}
        for sym in entered:
            for key, data in symptom_map.items():
                if key in sym:
                    result[key] = data
        if not result:
            return 'No matches found. Describe symptoms more clearly or consult a healthcare professional.'
        lines = ['Symptom Analysis (Educational Only):\n']
        for sym, data in result.items():
            lines.append(f'  • {sym.capitalize()}')
            lines.append(f'    Possible: {", ".join(data["conditions"])}')
            lines.append(f'    Urgency: {data["urgency"]}')
        lines.append('\nEducational only. Always consult a licensed physician.')
        return '\n'.join(lines)

    @tool
    def drug_information(drug_name: str) -> str:
        """Get drug details: class, uses, dosage, side effects, contraindications, interactions."""
        drug_db = {
            'paracetamol': {'class': 'Analgesic/Antipyretic', 'uses': 'Fever, mild-moderate pain', 'dosage': '500-1000mg every 4-6h, max 4g/day', 'side_effects': 'Rare at normal doses; overdose causes liver damage', 'contraindications': 'Severe liver disease', 'interactions': 'Warfarin (high doses)'},
            'ibuprofen':   {'class': 'NSAID', 'uses': 'Pain, fever, inflammation', 'dosage': '200-400mg every 4-6h, max 1200mg/day OTC', 'side_effects': 'GI upset, ulcers, increased BP, kidney issues', 'contraindications': 'Peptic ulcer, kidney disease, pregnancy (3rd trimester)', 'interactions': 'Aspirin, warfarin, ACE inhibitors'},
            'amoxicillin': {'class': 'Penicillin Antibiotic', 'uses': 'Bacterial infections: respiratory, UTI, ear, skin', 'dosage': '250-500mg every 8h or 875mg every 12h', 'side_effects': 'Diarrhea, nausea, rash, yeast overgrowth', 'contraindications': 'Penicillin allergy', 'interactions': 'Warfarin, oral contraceptives'},
            'metformin':   {'class': 'Biguanide Antidiabetic', 'uses': 'Type 2 diabetes (first-line)', 'dosage': 'Start 500mg twice daily, max 2550mg/day', 'side_effects': 'GI upset (temporary); rare: lactic acidosis', 'contraindications': 'Kidney failure (eGFR<30), liver disease', 'interactions': 'Alcohol, iodinated contrast, cimetidine'},
            'aspirin':     {'class': 'Salicylate NSAID / Antiplatelet', 'uses': 'Pain, fever; low-dose cardiovascular protection', 'dosage': 'Pain: 325-650mg. Cardio: 75-100mg/day', 'side_effects': 'GI irritation, tinnitus (high dose), bleeding', 'contraindications': 'Children <16 (Reye syndrome), peptic ulcer', 'interactions': 'Warfarin, ibuprofen, SSRIs'},
            'insulin':     {'class': 'Hormone / Injectable Antidiabetic', 'uses': 'Type 1 DM (essential), Type 2 DM (refractory)', 'dosage': 'Individualized — medical supervision required', 'side_effects': 'Hypoglycemia, weight gain, injection site reactions', 'contraindications': 'Hypoglycemia', 'interactions': 'Beta-blockers, corticosteroids, alcohol'},
        }
        key = drug_name.lower().strip()
        matched = next((k for k in drug_db if k in key or key in k), None)
        if not matched:
            return f"No local data for '{drug_name}'. Try hybrid_rag_retriever or wikipedia_live_tool."
        i = drug_db[matched]
        return (f"{matched.capitalize()}\n"
                f"  Class: {i['class']}\n  Uses: {i['uses']}\n  Dosage: {i['dosage']}\n"
                f"  Side Effects: {i['side_effects']}\n  Contraindications: {i['contraindications']}\n"
                f"  Key Interactions: {i['interactions']}\n\nFollow your doctor's/pharmacist's instructions.")

    tools = [hybrid_rag_retriever, faiss_rag_retriever, bm25_rag_retriever,
             wikipedia_live_tool, symptom_checker, drug_information]

    # ── LangGraph agent ──────────────────────────────────────────────────────
    SYSTEM_PROMPT = """You are MediBot, an expert Medical AI Assistant with a Hybrid RAG system.

## Tools Available:
1. hybrid_rag_retriever — PRIMARY: FAISS semantic + BM25 keyword. Use FIRST for all medical questions.
2. faiss_rag_retriever — Dense semantic search. Use for conceptual questions.
3. bm25_rag_retriever — Keyword search. Use for exact medical terms/drug names.
4. wikipedia_live_tool — Live Wikipedia. Use as fallback for rare/recent conditions.
5. symptom_checker — ALWAYS call when the user mentions any symptoms.
6. drug_information — ALWAYS call when a specific drug/medication is named.

## Rules:
- Start every medical question with hybrid_rag_retriever.
- Call symptom_checker immediately if symptoms are described.
- Call drug_information when a drug is named.
- Use wikipedia_live_tool for extra depth or unknown topics.
- Structure answers with headers and bullet points.
- Always end with a '🔧 Sources & Tools Used' section listing each tool invoked.
- Finish with the medical disclaimer.

*Medical Disclaimer: For educational purposes only. Always consult a qualified healthcare professional.*"""

    class AgentState(TypedDict):
        messages:   Annotated[list, add_messages]
        tools_used: List[str]

    llm = ChatGroq(model='llama-3.3-70b-versatile', api_key=groq_api_key,
                   temperature=0.2, max_tokens=1024)
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def call_model(state: AgentState):
        msgs = list(state['messages'])
        if not any(isinstance(m, SystemMessage) for m in msgs):
            msgs = [SystemMessage(content=SYSTEM_PROMPT)] + msgs
        return {'messages': [llm_with_tools.invoke(msgs)]}

    def run_tools(state: AgentState):
        last = state['messages'][-1]
        used = list(state.get('tools_used', []))
        if hasattr(last, 'tool_calls') and last.tool_calls:
            for tc in last.tool_calls:
                name = tc.get('name', 'unknown')
                if name not in used:
                    used.append(name)
        result = tool_node.invoke(state)
        result['tools_used'] = used
        return result

    def should_continue(state: AgentState):
        last = state['messages'][-1]
        return 'tools' if (hasattr(last, 'tool_calls') and last.tool_calls) else END

    graph = StateGraph(AgentState)
    graph.add_node('agent', call_model)
    graph.add_node('tools', run_tools)
    graph.set_entry_point('agent')
    graph.add_conditional_edges('agent', should_continue, {'tools': 'tools', END: END})
    graph.add_edge('tools', 'agent')
    agent = graph.compile()

    TOOL_LABELS = {
        'hybrid_rag_retriever': 'Hybrid RAG (FAISS + BM25)',
        'faiss_rag_retriever':  'FAISS Semantic RAG',
        'bm25_rag_retriever':   'BM25 Keyword RAG',
        'wikipedia_live_tool':  'Live Wikipedia',
        'symptom_checker':      'Symptom Checker',
        'drug_information':     'Drug Information DB',
    }
    TOOL_DESCRIPTIONS = {
        'hybrid_rag_retriever': 'FAISS semantic + BM25 keyword via Reciprocal Rank Fusion',
        'faiss_rag_retriever':  'Dense vector similarity — local Wikipedia KB',
        'bm25_rag_retriever':   'TF-IDF keyword matching — local Wikipedia KB',
        'wikipedia_live_tool':  'Real-time Wikipedia API lookup',
        'symptom_checker':      'Rule-based symptom-to-condition mapper with urgency levels',
        'drug_information':     'Local drug DB: dosage, side effects, contraindications, interactions',
    }

    return agent, TOOL_LABELS, TOOL_DESCRIPTIONS, HumanMessage, AIMessage, SystemMessage


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/hospital-emoji.png", width=72)
    st.title("MediBot")
    st.caption("Medical RAG Chatbot · Groq · LangGraph")
    st.divider()

    groq_key = st.text_input("🔑 Groq API Key",
                              type="password",
                              placeholder="gsk_...",
                              help="Get a free key at https://console.groq.com")

    st.divider()
    st.markdown("**🔧 Stack**")
    st.markdown("""
- 🤖 LLM: `Llama-3.3-70b` (Groq)
- 🧠 Embeddings: `all-MiniLM-L6-v2`
- 📚 Vector DB: FAISS + BM25
- 🌐 Knowledge: Wikipedia (55 topics)
- 🔗 Orchestration: LangGraph ReAct
    """)

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**💡 Try asking:**")
    st.markdown("""
- What is diabetes mellitus?
- I have fever, cough and fatigue
- Side effects of ibuprofen?
- Tell me about the immune system
- What is Kawasaki disease?
    """)

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("## 🏥 MediBot — Medical RAG Chatbot")
st.caption("Hybrid RAG (FAISS + BM25) · 6 Tools · Groq Llama-3.3-70b · Tool Attribution")
st.warning("⚠️ For educational purposes only. Always consult a qualified healthcare professional.", icon="⚕️")

# Guard: key required
if not groq_key:
    st.info("👈 Enter your Groq API key in the sidebar to get started.", icon="🔑")
    st.stop()

# Initialise pipeline (cached after first run)
try:
    agent, TOOL_LABELS, TOOL_DESCRIPTIONS, HumanMessage, AIMessage, SystemMessage = initialise(groq_key)
except Exception as e:
    st.error(f"Initialisation failed: {e}")
    st.stop()

# ── Chat state ────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # list of {"role", "content", "tools_used"}

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🏥"):
        st.markdown(msg["content"])
        if msg.get("tools_used"):
            with st.expander("🔧 RAG Tools Used", expanded=False):
                for t in msg["tools_used"]:
                    st.markdown(f"✅ **{TOOL_LABELS.get(t, t)}**  \n&nbsp;&nbsp;&nbsp;_{TOOL_DESCRIPTIONS.get(t, '')}_")

# ── Input ─────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a medical question…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt, "tools_used": []})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    # Build LangGraph history
    lc_history = []
    for m in st.session_state.messages[:-1]:   # exclude the new message (added below)
        if m["role"] == "user":
            lc_history.append(HumanMessage(content=m["content"]))
        else:
            lc_history.append(AIMessage(content=m["content"]))
    lc_history.append(HumanMessage(content=prompt))

    # Run agent
    with st.chat_message("assistant", avatar="🏥"):
        with st.spinner("🤔 Thinking…"):
            try:
                result = agent.invoke(
                    {"messages": lc_history, "tools_used": []},
                    config={"recursion_limit": 25},
                )
                ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
                reply = ai_msgs[-1].content if ai_msgs else "⚠️ No response received."
                tools_used = result.get("tools_used", [])
            except Exception as e:
                reply = f"⚠️ Agent error: {e}"
                tools_used = []

        st.markdown(reply)
        if tools_used:
            with st.expander("🔧 RAG Tools Used", expanded=True):
                for t in tools_used:
                    st.markdown(f"✅ **{TOOL_LABELS.get(t, t)}**  \n&nbsp;&nbsp;&nbsp;_{TOOL_DESCRIPTIONS.get(t, '')}_")

    st.session_state.messages.append({"role": "assistant", "content": reply, "tools_used": tools_used})
