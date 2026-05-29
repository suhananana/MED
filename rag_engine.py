import io
import re
import requests
import numpy as np
import faiss
import pdfplumber
import cohere

from bs4 import BeautifulSoup
from groq import Groq
from tavily import TavilyClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from Bio import Entrez

try:
    import docx
except:
    pass


class MedicalRAGEngine:

    def __init__(
        self,
        groq_api_key,
        tavily_api_key=None,
        cohere_api_key=None
    ):

        self.groq = Groq(api_key=groq_api_key)

        self.tavily = None
        if tavily_api_key:
            self.tavily = TavilyClient(api_key=tavily_api_key)

        self.cohere_client = None
        if cohere_api_key:
            self.cohere_client = cohere.Client(cohere_api_key)

        self.embedder = SentenceTransformer(
            "BAAI/bge-small-en-v1.5"
        )

        self.dimension = self.embedder.get_sentence_embedding_dimension()

        self.index = faiss.IndexFlatIP(self.dimension)

        self.chunks = []
        self.sources = []

        self.bm25_corpus = []
        self.bm25 = None

    # ---------------------------------
    # CHUNKING
    # ---------------------------------
    def chunk_text(self, text, chunk_size=500, overlap=100):

        words = text.split()

        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append(chunk)

        return chunks

    # ---------------------------------
    # EMBEDDINGS
    # ---------------------------------
    def embed(self, texts):

        vectors = self.embedder.encode(
            texts,
            normalize_embeddings=True
        )

        return np.array(vectors).astype("float32")

    # ---------------------------------
    # ADD CHUNKS
    # ---------------------------------
    def add_chunks(self, chunks, source):

        vectors = self.embed(chunks)

        self.index.add(vectors)

        self.chunks.extend(chunks)

        self.sources.extend([source] * len(chunks))

        tokenized = [c.split() for c in chunks]

        self.bm25_corpus.extend(tokenized)

        self.bm25 = BM25Okapi(self.bm25_corpus)

    # ---------------------------------
    # ADD TEXT
    # ---------------------------------
    def add_text(self, text, label="TEXT"):

        chunks = self.chunk_text(text)

        self.add_chunks(chunks, label)

    # ---------------------------------
    # ADD PDF
    # ---------------------------------
    def add_pdf(self, file_bytes, label="PDF"):

        text_data = []

        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:

            for page in pdf.pages:
                txt = page.extract_text()

                if txt:
                    text_data.append(txt)

        full_text = "\n".join(text_data)

        self.add_text(full_text, label)

    # ---------------------------------
    # ADD DOCX
    # ---------------------------------
    def add_docx(self, file_bytes, label="DOCX"):

        document = docx.Document(io.BytesIO(file_bytes))

        paragraphs = []

        for p in document.paragraphs:
            if p.text.strip():
                paragraphs.append(p.text)

        text = "\n".join(paragraphs)

        self.add_text(text, label)

    # ---------------------------------
    # WEB SCRAPER
    # ---------------------------------
    def scrape_medical_site(self, url):

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, headers=headers)

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav"]):
            tag.decompose()

        text = soup.get_text(separator=" ")

        return re.sub(r"\s+", " ", text)

    # ---------------------------------
    # FAISS RETRIEVAL
    # ---------------------------------
    def faiss_search(self, query, top_k=4):

        q_vec = self.embed([query])

        scores, indices = self.index.search(q_vec, top_k)

        results = []

        for idx in indices[0]:
            if idx >= 0:
                results.append(self.chunks[idx])

        return results

    # ---------------------------------
    # BM25 RETRIEVAL
    # ---------------------------------
    def bm25_search(self, query, top_k=4):

        tokenized_query = query.split()

        scores = self.bm25.get_scores(tokenized_query)

        ranked = np.argsort(scores)[::-1][:top_k]

        results = [self.chunks[i] for i in ranked]

        return results

    # ---------------------------------
    # PUBMED SEARCH
    # ---------------------------------
    def pubmed_search(self, query):

        Entrez.email = "your_email@gmail.com"

        handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=3
        )

        record = Entrez.read(handle)

        ids = record["IdList"]

        if not ids:
            return []

        fetch = Entrez.efetch(
            db="pubmed",
            id=ids,
            rettype="abstract",
            retmode="text"
        )

        text = fetch.read()

        return [text]

    # ---------------------------------
    # WEB SEARCH
    # ---------------------------------
    def web_search(self, query):

        if not self.tavily:
            return []

        medical_query = (
            f"site:nih.gov OR site:who.int OR site:cdc.gov {query}"
        )

        results = self.tavily.search(
            query=medical_query,
            search_depth="advanced",
            max_results=5
        )

        chunks = []

        for r in results["results"]:
            chunks.append(r["content"])

        return chunks

    # ---------------------------------
    # RERANKER
    # ---------------------------------
    def rerank(self, query, docs):

        if not self.cohere_client:
            return docs[:4]

        rerank = self.cohere_client.rerank(
            query=query,
            documents=docs,
            top_n=4,
            model="rerank-english-v3.0"
        )

        ranked_docs = []

        for item in rerank.results:
            ranked_docs.append(docs[item.index])

        return ranked_docs

    # ---------------------------------
    # MAIN QUERY
    # ---------------------------------
    def query(self, question):

        tools_used = []

        docs = []

        if len(self.chunks) > 0:
            faiss_docs = self.faiss_search(question)
            docs.extend(faiss_docs)
            tools_used.append("🧠 FAISS Dense Retrieval")

        if self.bm25:
            bm25_docs = self.bm25_search(question)
            docs.extend(bm25_docs)
            tools_used.append("📚 BM25 Sparse Retrieval")

        if any(word in question.lower() for word in [
            "research",
            "study",
            "paper",
            "clinical"
        ]):
            pubmed_docs = self.pubmed_search(question)
            docs.extend(pubmed_docs)
            tools_used.append("📄 PubMed Retrieval")

        if self.tavily:
            web_docs = self.web_search(question)
            docs.extend(web_docs)
            tools_used.append("🌐 Live Medical Web Search")

        docs = self.rerank(question, docs)

        tools_used.append("🔍 Cohere Re-ranking")

        context = "\n\n".join(docs[:6])

        system_prompt = f"""
You are an advanced AI medical assistant.

Use ONLY the provided medical context.

If unsure, clearly say the information may require professional medical consultation.

Context:
{context}
"""

        response = self.groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=0.3,
            max_tokens=1024
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "tools_used": tools_used,
            "sources": self.sources[:5]
        }
