import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI

from tools.rag_tool import medical_rag_search
from tools.scraper_tool import scrape_medical_web
from tools.symptom_checker import check_symptoms
from tools.medicine_tool import medicine_info
from tools.wiki_tool import medical_wikipedia_search

st.set_page_config(page_title="MediBot 🏥", page_icon="🏥", layout="wide")

st.title("🏥 MediBot — Medical AI Assistant")

GOOGLE_API_KEY = st.sidebar.text_input("Enter Gemini API Key", type="password")

if not GOOGLE_API_KEY:
    st.warning("Please enter Gemini API key")
    st.stop()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)

TOOLS = [
    Tool(
        name="medical_rag_retriever",
        func=medical_rag_search,
        description="Searches medical vector database"
    ),
    Tool(
        name="medical_web_scraper",
        func=scrape_medical_web,
        description="Scrapes medical information from web"
    ),
    Tool(
        name="symptom_checker",
        func=check_symptoms,
        description="Checks symptoms and possible conditions"
    ),
    Tool(
        name="medicine_information",
        func=medicine_info,
        description="Provides medicine information"
    ),
    Tool(
        name="medical_wikipedia",
        func=medical_wikipedia_search,
        description="Search medical topics from Wikipedia"
    )
]

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

agent = initialize_agent(
    tools=TOOLS,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state.messages = []

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
            try:
                response = agent.run(prompt)

                final_response = f"""
{response}

⚠️ Disclaimer:
This chatbot is for educational purposes only.
Please consult a qualified doctor for medical advice.
"""

                st.markdown(final_response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_response
                })

            except Exception as e:
                st.error(f"Error: {str(e)}")
