# 🤖 ResumePulse AI — Smart Resume Screening & ATS Screener

**ResumePulse AI** is an AI-powered recruitment platform and Advanced ATS (Applicant Tracking System) powered by **Groq AI (Llama 3)** to automate resume screening, calculate ATS scores, extract skills, match candidates with job descriptions, detect skill gaps and red flags, verify authenticity, and provide intelligent hiring recommendations to help recruiters make faster and smarter hiring decisions.

---

## 🌟 Key Features

### 1️⃣ Smart Resume Parsing & Ranking
- **Multi-Format Support**: Upload resumes in **PDF** or **DOCX** formats.
- **Intelligent Matching**: Uses **TF-IDF & Cosine Similarity** for speed, or **AI Semantic Analysis (Groq Llama 3)** for deep contextual matching.
- **Skill Gap Analysis**: Identifies missing skills and strengths for each candidate.

### 2️⃣ Advanced AI Analysis
- **🤖 Smart Resume Chatbot**: Chat with any candidate's resume using **Llama 3 via Groq**. Ask questions like *"Is this candidate good for a Senior role?"* or *"Summarize their experience"*.
- **🔮 Hiring Prediction**: AI estimates the **Hiring Probability** (High/Medium/Low) based on the match score and skills.
- **🕵️‍♂️ Authenticity Check**: Automatically detects red flags like **keyword stuffing**, suspicious length, or missing contact info to prevent fraud.
- **Bias-Free Screening**: Toggle "Bias-Free Mode" to redact PII (Name, Email, Phone) for fair evaluation.

### 3️⃣ Productivity Tools
- **📧 Automated Email Outreach**: Generate personalized **Interview Invitation** or **Rejection Emails** with one click.
- **🎤 Interview Question Generator**: Automatically creates technical questions tailored to the candidate's specific skill set.

### 4️⃣ Visual Dashboard
- **Interactive Rankings**: Sort and filter candidates by match score.
- **Comparative Analysis**: Compare multiple candidates side-by-side.
- **Data Visualization**: interactive charts using **Plotly**.

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **AI / LLM**: **Groq API** (Llama 3.3 70B Versatile)
- **Backend Logic**: Python
- **NLP & ML**: Scikit-Learn, Spacy
- **Data Processing**: PyPDF2, python-docx, Pandas

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+ installed
- A **Groq API Key** (Free tier available at [console.groq.com](https://console.groq.com))

### 1. Clone the Repository
```bash
git clone https://github.com/vaishnavireddy067/ResumePulse-AI.git
cd ResumePulse-AI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up API Key
To enable the AI Chatbot and Semantic Ranking, set your Groq API Key.

**Windows PowerShell:**
```powershell
$env:GROQ_API_KEY="your_actual_api_key_here"
```

**Mac/Linux:**
```bash
export GROQ_API_KEY="your_actual_api_key_here"
```

### 4. Run the Application
```bash
streamlit run app.py
```
*Note: If `streamlit` command is not found, try `python -m streamlit run app.py`*

The app will open in your browser at `http://localhost:8501`.

---

## 📖 Usage Guide

1. **Enter Job Description**: Paste the JD in the left sidebar.
2. **Upload Resumes**: Select multiple PDF/DOCX files.
3. **Choose Ranking Method**: Select **"AI Semantic (Groq Llama 3)"** for best accuracy.
4. **Analyze**: Click "Analyze & Rank Candidates".
5. **Insights**:
   - Check the **Hiring Prediction Badge** (Green/Orange/Red).
   - Look for **Authenticity Warnings** ⚠️.
   - Use the **"Chat with Resume"** tab to ask specific questions.
   - Go to **"📧 Outreach"** tab to copy a generated email.

---

## 📂 Project Structure

```
HR_Resume_Screening_AI/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Dependencies
├── utils/
│   ├── parser.py          # Text extraction
│   ├── matcher.py         # Ranking algorithms
│   ├── extractor.py       # Entity extraction (Name, Email)
│   ├── ai_features.py     # AI logic (Prediction, Auth Check, Email)
│   ├── rag.py             # RAG / Chatbot logic
│   └── translator.py      # Translation
└── README.md              # Documentation
```

---

## 🔮 Future Roadmap

- [ ] **Career Trajectory Analysis**: Visualize candidate growth over time.
- [ ] **Social Media Scanning**: (Optional) Check LinkedIn profiles.
- [ ] **Database Integration**: Save candidates to a database (SQL/NoSQL).

#
