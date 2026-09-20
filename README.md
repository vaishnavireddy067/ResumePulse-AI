# 🧠 HireMind AI — Agentic Talent Intelligence Platform

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Google Gemini](https://img.shields.io/badge/AI-Google%20Gemini-4285F4.svg)](https://ai.google.dev/)
[![Groq](https://img.shields.io/badge/AI-Groq%20Llama%203.3-F55036.svg)](https://groq.com/)
[![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-0052CC.svg)](https://github.com/facebookresearch/faiss)

**HireMind AI** is an Agentic AI-powered recruitment and talent intelligence platform that automates resume screening, deep ATS scoring, skill gap matching, customized coding assessment generation, structured interview rubrics, and recruiter outreach using collaborative AI agents.

---

## 🌟 Key Capabilities & Collaborative Agents

```
                               ┌────────────────────────────────┐
                               │     Recruiter / Hiring Team    │
                               └───────────────┬────────────────┘
                                               │ Uploads JDs & Resumes
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                             🧠 HIREMIND MULTI-AGENT ORCHESTRATOR                            │
├───────────────────┬───────────────────┬────────────────────┬────────────────────────────────┤
│ 🕵️ Screening Agent│ 🎯 SkillGap Agent │ 💻 Coding Agent    │ 🎙️ Interview Rubric Agent       │
│ • Parsing PDF/DOCX│ • Core Match      │ • Tailored Test    │ • Deep Technical Questions     │
│ • Blended ATS Sc. │ • Missing Skills  │ • Boilerplate Code │ • Behavioral Scenarios         │
│ • Semantic AI Fit │ • Growth Areas    │ • Rubric & Tests   │ • Architecture Challenges      │
├───────────────────┴───────────────────┴────────────────────┴────────────────────────────────┤
│ 🛡️ Red-Flag & Authenticity Agent   │ 💬 Vector RAG Chatbot    │ 📧 Automated Outreach Agent │
│ • Keyword-stuffing detection       │ • FAISS Q&A with Resumes │ • 1-Click Interview Invites │
│ • PII Redaction (Bias-Free Mode)   │ • Interactive Memory     │ • Rejections & Offer Notes  │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1️⃣ Intelligent Resume Screening & ATS Ranking
- **Multi-Format Extraction**: Batch upload and parse **PDF** and **DOCX** candidate resumes.
- **Hybrid Scoring**: Combines **TF-IDF Keyword Density** and **Semantic AI Embeddings (Gemini / Groq Llama 3.3)** for fair, deep ranking.
- **Hiring Probability Prediction**: AI badges (🟢 High, 🟡 Potential, 🔴 Low) with concrete next-step recommendations.

### 2️⃣ Tailored Coding Assessment Studio
- Dynamically creates customized live coding challenges matching the candidate's exact tech stack and seniority level (Junior, Mid-Level, Senior, Lead).
- Provides starter boilerplate code, sample test cases, and multi-criteria evaluation rubrics.

### 3️⃣ Structured Interview & System Design Rubrics
- Generates deep technical questions with **"Good Answer Signals"** and **"Red Flag Warnings"**.
- STAR-method behavioral questions and real-world architectural scenarios.

### 4️⃣ Authenticity & Bias-Free Screening
- **Fraud & Anomaly Detection**: Identifies keyword stuffing, suspicious brevity, and missing contact information.
- **Bias-Free Mode**: Automatically redacts Personally Identifiable Information (PII) including Name, Email, and Phone number.

### 5️⃣ Vector RAG Resume Chatbot
- Chat interactively with any candidate's resume using **FAISS** vector embeddings and conversational LLM reasoning.

### 6️⃣ Talent Analytics & Automated Outreach
- Interactive Plotly visualizations for candidate score distribution and talent fit.
- One-click personalized email generation for interview invites, offer discussions, and polite rejections.
- CSV / Excel export for ATS integration.

---

## 🛠️ Tech Stack

- **Frontend & Dashboard**: Streamlit (Modern Custom Themed UI)
- **AI / LLMs**: Google Gemini (`google-generativeai`) & Groq (`llama-3.3-70b-versatile`)
- **Vector Search & RAG**: FAISS, LangChain, Sentence-Transformers
- **NLP & Parsing**: PyPDF2, python-docx, Scikit-Learn, Spacy
- **Data Analytics & Charts**: Pandas, Plotly

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/vaishnavireddy067/HireMind-AI.git
cd HireMind-AI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys (Optional but Recommended)
Set your Google Gemini or Groq API key in your environment or enter it directly in the application sidebar:

**Windows PowerShell:**
```powershell
$env:GEMINI_API_KEY="your_gemini_api_key"
$env:GROQ_API_KEY="your_groq_api_key"
```

**macOS / Linux:**
```bash
export GEMINI_API_KEY="your_gemini_api_key"
export GROQ_API_KEY="your_groq_api_key"
```

### 4. Run the Application
```bash
streamlit run app.py
```

The application will launch in your browser at `http://localhost:8501`.

---

## 📂 Project Structure

```
HireMind-AI/
├── app.py                      # Main Streamlit Talent Intelligence Dashboard
├── requirements.txt            # Project dependencies
├── utils/
│   ├── agentic_pipeline.py     # Multi-Agent Collaborative Orchestrator
│   ├── ai_features.py          # Gemini/Groq LLM Engine & Assessment Generators
│   ├── rag.py                  # FAISS Vector Store & RAG Chatbot
│   ├── parser.py               # PDF & DOCX text extraction
│   ├── matcher.py              # TF-IDF & Cosine Similarity matching
│   ├── extractor.py            # Regex & Contact extraction
│   └── translator.py           # Multi-language resume translator
└── README.md                   # Platform documentation
```

---

## 📄 License
This project is licensed under the MIT License.
