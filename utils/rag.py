"""
HireMind AI — Resume RAG & Conversational Intelligence Engine
Retrieval-Augmented Generation for natural language Q&A with candidate resumes.
Supports FAISS vector embeddings, Google Gemini, and Groq LLMs.
"""

import os
import re
from typing import Dict, Any, List, Optional
from utils.ai_features import get_llm_response

# Optional FAISS & Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


class ResumeVectorStore:
    def __init__(self):
        self.encoder = None
        self.index = None
        self.chunks = []
        self.raw_text = ""
        
        if EMBEDDINGS_AVAILABLE:
            try:
                self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self.encoder = None

    def build_from_text(self, text: str, chunk_size: int = 500, overlap: int = 100):
        self.raw_text = text
        self.chunks = self._create_chunks(text, chunk_size, overlap)
        
        if self.encoder and len(self.chunks) > 0:
            try:
                embeddings = self.encoder.encode(self.chunks, show_progress_bar=False)
                dim = embeddings.shape[1]
                self.index = faiss.IndexFlatL2(dim)
                self.index.add(np.array(embeddings).astype("float32"))
            except Exception as e:
                print(f"FAISS indexing note: {e}")
                self.index = None

    def _create_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += max(chunk_size - overlap, 50)
        return chunks

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if self.index and self.encoder:
            try:
                query_vec = self.encoder.encode([query])
                distances, indices = self.index.search(np.array(query_vec).astype("float32"), min(top_k, len(self.chunks)))
                return [self.chunks[idx] for idx in indices[0] if idx < len(self.chunks)]
            except Exception:
                pass

        # Fallback heuristic retrieval
        query_words = set(re.findall(r'\w+', query.lower()))
        scored = []
        for chunk in self.chunks:
            chunk_words = set(re.findall(r'\w+', chunk.lower()))
            score = len(query_words.intersection(chunk_words))
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for s, c in scored[:top_k]] if scored else self.chunks[:top_k]


def create_resume_vector_db(text: str, candidate_id: Optional[str] = None) -> ResumeVectorStore:
    """Initializes and builds a vector store for a candidate's resume."""
    store = ResumeVectorStore()
    store.build_from_text(text)
    return store


def query_resume(vector_store: Any, query: str, candidate_name: str = "Candidate", provider: str = "auto") -> str:
    """
    RAG-powered interactive Q&A against candidate resume.
    """
    if not vector_store:
        return "❌ No resume context found. Please upload or select a candidate first."

    # Retrieve context chunks
    if isinstance(vector_store, ResumeVectorStore):
        relevant_chunks = vector_store.retrieve(query, top_k=3)
        context = "\n---\n".join(relevant_chunks)
    elif isinstance(vector_store, dict):
        chunks = vector_store.get("chunks", [])
        context = "\n---\n".join(chunks[:3])
    else:
        context = str(vector_store)[:2000]

    prompt = f"""
    You are HireMind AI's Talent Intelligence Assistant.
    You are answering a recruiter's question about candidate: '{candidate_name}'.

    RELEVANT RESUME EXCERPTS:
    {context}

    RECRUITER QUESTION:
    {query}

    INSTRUCTIONS:
    - Answer directly, factually, and concisely using the candidate's resume data.
    - If the resume explicitly mentions specific numbers, tools, or projects, highlight them.
    - If information is not in the resume, clearly state: "The resume does not explicitly mention this, but..."
    - Maintain an executive, objective tone suitable for hiring managers.
    """

    system_prompt = "You are a professional HR AI assistant specializing in candidate resume analysis."
    ai_answer = get_llm_response(prompt, provider=provider, system_prompt=system_prompt)

    if ai_answer:
        return ai_answer

    # Fallback response if no LLM keys are configured
    return f"""### 📄 Relevant Resume Excerpt:
{context}

---
*💡 Note: Connect a Gemini API Key or Groq API Key in the sidebar to enable live natural language AI reasoning!*"""
