import streamlit as st
from rag_engine import MedicalRAGEngine

st.set_page_config(
    page_title="Medical Agentic RAG",
    page_icon="🩺",
    layout="wide"
)

# ----------------------------
# API KEYS
# ----------------------------
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
TAVILY_API_KEY = "YOUR_TAVILY_API_KEY"
COHERE_API_KEY = "YOUR_COHERE_API_KEY"

# ----------------------------
# SESSION
# ----------------------------
if "rag" not in st.session_state:
    st.session_state.rag = MedicalRAGEngine(
        groq_api_key=GROQ_API_KEY,
        tavily_api_key=TAVILY_API_KEY,
        cohere_api_key=COHERE_API_KEY
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# SIDEBAR
# ----------------------------
with st.sidebar:
    st.title("🩺 Medical Agentic RAG")

    st.markdown("### Upload Medical Files")

    uploaded_file = st.file_uploader(
        "Upload PDF/DOCX/TXT",
        type=["pdf", "docx", "txt"]
    )

    if uploaded_file:

        if st.button("Index File"):

            file_bytes = uploaded_file.read()

            if uploaded_file.name.endswith(".pdf"):
                st.session_state.rag.add_pdf(file_bytes, uploaded_file.name)

            elif uploaded_file.name.endswith(".docx"):
                st.session_state.rag.add_docx(file_bytes, uploaded_file.name)

            elif uploaded_file.name.endswith(".txt"):
                text = file_bytes.decode("utf-8")
                st.session_state.rag.add_text(text, uploaded_file.name)

            st.success("File Indexed Successfully")

# ----------------------------
# MAIN UI
# ----------------------------
st.title("💬 Medical AI Assistant")

st.markdown("""
This chatbot uses:
- FAISS Dense Retrieval
- BM25 Sparse Retrieval
- PubMed Retrieval
- Tavily Web Search
- Cohere Re-ranking
- Groq LLM
""")

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask medical questions...")

if prompt:

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            result = st.session_state.rag.query(prompt)

            answer = result["answer"]
            tools = result["tools_used"]
            sources = result["sources"]

            st.markdown(answer)

            st.markdown("---")

            st.subheader("🧠 Tools Used")

            for tool in tools:
                st.success(tool)

            st.subheader("📚 Sources")

            for src in sources:
                st.info(src)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })
