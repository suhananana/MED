from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

MEDICAL_TOPICS = [
    "Diabetes",
    "Hypertension",
    "Asthma",
    "COVID-19",
    "Tuberculosis",
    "Cancer"
]

all_docs = []

for topic in MEDICAL_TOPICS:
    loader = WikipediaLoader(query=topic, load_max_docs=2)
    docs = loader.load()
    all_docs.extend(docs)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(all_docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)

vectorstore.save_local("vector_db")

print("FAISS DB Created")
