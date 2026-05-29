from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "vector_db",
    EMBEDDING_MODEL,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4}
)

def medical_rag_search(query: str):
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant medical documents found."

    return "\n\n".join([doc.page_content for doc in docs])
