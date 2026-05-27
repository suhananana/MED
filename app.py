import streamlit as st
from groq import Groq

# ── Page config ─────────────────────────────────────────
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered",
)

# ── Custom CSS (dark, ChatGPT-like) ──────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #212121;
    color: #ececec;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Chat messages */
.stChatMessage {
    background: transparent !important;
    border: none !important;
}

/* User bubble */
[data-testid="stChatMessageContent"] {
    background: #2f2f2f !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    color: #ececec !important;
}

/* Input box */
.stChatInputContainer {
    background: #2f2f2f !important;
    border-radius: 16px !important;
    border: 1px solid #444 !important;
    padding: 4px 8px !important;
}

textarea {
    background: transparent !important;
    color: #ececec !important;
}

/* Title */
h1 { color: #ececec !important; text-align: center; }
p  { color: #aaa !important; text-align: center; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #171717 !important;
}
section[data-testid="stSidebar"] * {
    color: #ccc !important;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar: API key + settings ──────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    api_key = st.text_input("Groq API Key", type="password",
                            placeholder="gsk_...",
                            help="Free key at https://console.groq.com")
    
    model = st.selectbox("Model", [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ])
    
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
    
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful, knowledgeable AI assistant. Answer questions clearly and accurately.",
        height=100
    )
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("**Get free API key:**\nhttps://console.groq.com")

# ── Title ────────────────────────────────────────────────
st.title("🤖 AI Chatbot")
st.markdown("Ask me anything — I'm powered by Groq's ultra-fast LLMs.")

# ── Session state ────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Render history ───────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input ────────────────────────────────────────────────
user_input = st.chat_input("Message AI Chatbot...")

if user_input:
    if not api_key:
        st.error("⚠️ Please enter your Groq API key in the sidebar.")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get response
    client = Groq(api_key=api_key)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        *[{"role": m["role"], "content": m["content"]}
                          for m in st.session_state.messages]
                    ],
                    temperature=temperature,
                    max_tokens=2048,
                    stream=True,
                )
                
                full_reply = ""
                placeholder = st.empty()
                for chunk in response:
                    delta = chunk.choices[0].delta.content or ""
                    full_reply += delta
                    placeholder.markdown(full_reply + "▌")
                placeholder.markdown(full_reply)
                
            except Exception as e:
                full_reply = f"❌ Error: {e}"
                st.error(full_reply)

    st.session_state.messages.append({"role": "assistant", "content": full_reply})
