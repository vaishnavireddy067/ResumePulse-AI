"""
HireMind AI — Advanced AI & LLM Engine
Supports Google Gemini, Groq (Llama 3.3), and offline heuristic fallbacks.
"""

import os
import re
import random
import json
from typing import Dict, Any, List, Optional

# Optional Spacy import
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = spacy.blank("en")
except ImportError:
    nlp = None

# Optional Google Gemini import
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Optional Groq import
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


def get_llm_response(prompt: str, provider: str = "auto", system_prompt: Optional[str] = None) -> str:
    """
    Unified LLM call supporting Google Gemini & Groq with automatic fallback.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")

    full_prompt = prompt
    if system_prompt:
        full_prompt = f"System: {system_prompt}\n\nTask:\n{prompt}"

    # 1. Try Gemini if selected or auto
    if provider in ["gemini", "auto"] and GEMINI_AVAILABLE and gemini_key:
        try:
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(full_prompt)
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            print(f"Gemini LLM error: {e}")
            if provider == "gemini":
                return f"Gemini Error: {e}"

    # 2. Try Groq if selected or auto
    if provider in ["groq", "auto"] and GROQ_AVAILABLE and groq_key:
        try:
            client = Groq(api_key=groq_key)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            completion = client.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile",
                temperature=0.3
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Groq LLM error: {e}")
            if provider == "groq":
                return f"Groq Error: {e}"

    # 3. Fallback Heuristic response
    return ""


def semantic_match(jd: str, resume_text: str, provider: str = "auto") -> float:
    """
    Computes semantic match score (0-100) using Gemini or Groq.
    """
    prompt = f"""
    Act as an expert Applicant Tracking System (ATS).
    Evaluate the match between this Candidate Resume and the Job Description (JD).
    
    Job Description:
    {jd[:1500]}
    
    Candidate Resume:
    {resume_text[:1500]}
    
    Give a strict, objective match score from 0 to 100 based on skills, experience relevance, and requirements.
    Return ONLY an integer number (e.g. 85).
    """
    
    response = get_llm_response(prompt, provider=provider, system_prompt="You are a strict ATS scoring engine. Output only a single number between 0 and 100.")
    
    if response:
        match = re.search(r'\b\d{1,3}\b', response)
        if match:
            val = float(match.group())
            return min(max(val, 0.0), 100.0)

    # Heuristic Jaccard Fallback
    jd_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', jd.lower()))
    resume_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', resume_text.lower()))
    if not jd_words:
        return 0.0
    common = jd_words.intersection(resume_words)
    return round(min(len(common) / max(len(jd_words) * 0.4, 1.0) * 100, 100.0), 1)


def generate_coding_assessment(jd: str, resume_text: str, seniority: str = "Mid-Level", provider: str = "auto") -> Dict[str, Any]:
    """
    Generates a customized live coding challenge based on JD and Candidate profile.
    """
    prompt = f"""
    You are an expert Principal Engineer and Technical Interview Architect.
    Design a tailored coding assessment problem for a {seniority} candidate based on the following:

    Job Description:
    {jd[:1000]}

    Candidate Background:
    {resume_text[:1000]}

    Provide your assessment in valid JSON with exactly these keys:
    - "title": "Problem Title",
    - "difficulty": "{seniority} (Easy/Medium/Hard)",
    - "estimated_time": "e.g. 45 minutes",
    - "core_technologies": ["Python", "FastAPI", etc.],
    - "problem_statement": "Clear detailed problem description with constraints",
    - "boilerplate_code": "Starting starter code template with function signature",
    - "sample_test_cases": [
        {{"input": "sample input", "expected_output": "sample output", "explanation": "why this output"}}
      ],
    - "evaluation_rubric": [
        {{"criterion": "Code Quality & Clean Architecture", "weight": "30%", "description": "Readable, modular code"}},
        {{"criterion": "Edge Case Handling & Correctness", "weight": "40%", "description": "Handles nulls, bounds"}},
        {{"criterion": "Time & Space Complexity", "weight": "30%", "description": "Optimal algorithmic efficiency"}}
      ]
    """
    response = get_llm_response(prompt, provider=provider, system_prompt="You are a senior tech lead. Return only valid JSON.")

    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass

    # High-quality fallback coding assessment
    return {
        "title": "Optimized Async Data Ingestion Pipeline",
        "difficulty": f"{seniority} (Medium)",
        "estimated_time": "45 minutes",
        "core_technologies": ["Python", "Data Structures", "AsyncIO"],
        "problem_statement": "Design and implement a thread-safe / async rate-limited batch processing function that ingests high-throughput data payloads, validates schemas, deduplicates items by unique ID, and outputs aggregate metrics in O(N) time.",
        "boilerplate_code": """def process_candidate_stream(records: list[dict], batch_size: int = 100) -> dict:
    \"\"\"
    Process and aggregate streaming records.
    :param records: List of raw telemetry/event payloads
    :param batch_size: Max items per processed chunk
    :return: Summary metrics dict
    \"\"\"
    # TODO: Implement deduplication and batch aggregation
    pass
""",
        "sample_test_cases": [
            {"input": "[{'id': 1, 'val': 10}, {'id': 1, 'val': 10}, {'id': 2, 'val': 25}]", "expected_output": "{'unique_count': 2, 'sum': 35}", "explanation": "Duplicate ID 1 is filtered."},
            {"input": "[]", "expected_output": "{'unique_count': 0, 'sum': 0}", "explanation": "Empty input handled cleanly."}
        ],
        "evaluation_rubric": [
            {"criterion": "Algorithmic Efficiency", "weight": "35%", "description": "Optimal O(N) time and O(N) space complexity."},
            {"criterion": "Code Modularity & Type Safety", "weight": "35%", "description": "Clean functions, docstrings, type annotations."},
            {"criterion": "Error Handling", "weight": "30%", "description": "Graceful handling of corrupt inputs and boundary conditions."}
        ]
    }


def generate_interview_rubric(jd: str, resume_text: str, provider: str = "auto") -> Dict[str, Any]:
    """
    Generates structured role-tailored technical & behavioral interview questions with answer rubrics.
    """
    prompt = f"""
    Act as a VP of Engineering and Lead Technical Recruiter.
    Create a comprehensive 4-stage interview evaluation rubric for this candidate:

    Job Description:
    {jd[:1000]}

    Candidate Resume:
    {resume_text[:1000]}

    Output strictly in JSON format with keys:
    - "technical_questions": [
        {{"question": "Deep technical question", "target_competency": "System Design / Core Tech", "good_answer_signals": "What a top candidate says", "red_flag_signals": "What a weak candidate says"}}
      ],
    - "behavioral_questions": [
        {{"question": "Behavioral / Leadership scenario", "target_competency": "Collaboration / Ownership", "good_answer_signals": "STAR method response", "red_flag_signals": "Blaming others"}}
      ],
    - "architecture_scenario": {{"scenario": "A real-world scaling or architectural challenge", "evaluation_points": ["Point 1", "Point 2"]}}
    """
    response = get_llm_response(prompt, provider=provider, system_prompt="You are a top-tier hiring manager. Output only valid JSON.")

    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass

    return {
        "technical_questions": [
            {
                "question": "Can you explain how you designed and optimized the most complex backend service listed on your resume?",
                "target_competency": "Architecture & Scaling",
                "good_answer_signals": "Articulates trade-offs, caching strategies, bottleneck resolution with concrete metrics.",
                "red_flag_signals": "Vague about tech choices, cannot explain failure modes or concurrency."
            },
            {
                "question": "How do you ensure data consistency and reliability across distributed microservices?",
                "target_competency": "Distributed Systems",
                "good_answer_signals": "Mentions idempotent consumers, saga patterns, event sourcing, or 2PC trade-offs.",
                "red_flag_signals": "Assumes database transactions work automatically across remote services."
            }
        ],
        "behavioral_questions": [
            {
                "question": "Tell me about a time when a critical bug occurred in production under your watch. How did you handle it?",
                "target_competency": "Ownership & Incident Response",
                "good_answer_signals": "Takes ownership, performs structured root-cause analysis (RCA), adds regression tests.",
                "red_flag_signals": "Deflects responsibility or lacks post-incident prevention mindset."
            }
        ],
        "architecture_scenario": {
            "scenario": "Design a high-concurrency API service that must handle 50,000 requests/sec with p99 latency under 40ms.",
            "evaluation_points": [
                "Layered caching (Redis / In-memory LRU)",
                "Asynchronous non-blocking I/O",
                "Load balancing and horizontal pod autoscaling",
                "Database connection pooling and read replicas"
            ]
        }
    }


def predict_hiring_chance(score: float, missing_skills_count: int, experience_years: Optional[float] = None) -> Dict[str, Any]:
    """
    Estimates Hiring Probability based on ATS match score and skill gaps.
    """
    adjusted_score = score - (missing_skills_count * 2.5)
    
    if adjusted_score >= 75:
        return {
            "label": "Strong Fit (High Probability)",
            "tier": "High",
            "color": "#10B981", # Emerald
            "badge": "🟢 Top Candidate",
            "recommendation": "Fast-track to Technical Interview"
        }
    elif adjusted_score >= 50:
        return {
            "label": "Potential Fit (Medium Probability)",
            "tier": "Medium",
            "color": "#F59E0B", # Amber
            "badge": "🟡 Potential Match",
            "recommendation": "Conduct Initial Phone Screen / Assessment"
        }
    else:
        return {
            "label": "Low Alignment (Low Probability)",
            "tier": "Low",
            "color": "#EF4444", # Red
            "badge": "🔴 Unlikely Match",
            "recommendation": "Keep on file or send polite rejection"
        }


def check_authenticity(resume_text: str) -> Dict[str, Any]:
    """
    Audits resume for keyword stuffing, repetitive anomalies, and missing contact information.
    """
    flags = []
    text_lower = resume_text.lower()
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text_lower)
    
    # 1. Check length
    if len(words) < 50:
        flags.append("Resume content is unusually short (< 50 words).")
    
    # 2. Check keyword repetition / stuffing
    if words:
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
        
        for w, count in word_counts.items():
            if count > 18 and count / len(words) > 0.07:
                flags.append(f"Potential keyword stuffing: word '{w}' appears {count} times ({round(count/len(words)*100, 1)}% frequency).")

    # 3. Check contact info presence
    has_email = bool(re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', resume_text))
    has_phone = bool(re.search(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', resume_text))

    if not has_email:
        flags.append("No valid email address detected.")
    if not has_phone:
        flags.append("No recognizable telephone contact detected.")

    status = "Passed" if len(flags) == 0 else "Review Needed" if len(flags) <= 2 else "High Risk"
    
    return {
        "status": status,
        "is_suspicious": len(flags) > 0,
        "flag_count": len(flags),
        "flags": flags if flags else ["No suspicious anomalies or keyword stuffing detected. Authenticity verified."]
    }


def generate_email(candidate_name: str, job_title: str, status: str = "interview", feedback: str = "", provider: str = "auto") -> str:
    """
    Generates personalized recruiter outreach emails.
    """
    prompt = f"""
    Act as a warm, professional Talent Acquisition Manager at a high-growth tech company.
    Write an email to candidate '{candidate_name}' for the position '{job_title}'.
    
    Email Type: {status.upper()} (interview invite / polite rejection / offer discussion)
    Specific Feedback/Notes: {feedback if feedback else 'Standard professional communication'}
    
    Format with:
    Subject Line: ...
    Body: ...
    """
    
    response = get_llm_response(prompt, provider=provider, system_prompt="You are a professional HR recruiter. Output clear, ready-to-send email copy.")
    if response:
        return response

    if status == "interview":
        return f"""Subject: Invitation to Interview: {job_title} at HireMind Team

Dear {candidate_name},

Thank you for your application for the {job_title} role. We were thoroughly impressed by your background and project experience.

We would love to invite you for a 45-minute technical conversation with our team to discuss your past achievements and learn more about your career goals.

Please let us know your availability over the coming days, or schedule directly using our calendar link.

Best regards,
Talent Acquisition Team
HireMind AI Platform"""
    elif status == "offer":
        return f"""Subject: Exciting News regarding your application for {job_title}!

Dear {candidate_name},

Following our recent interview discussions, the team has been unanimous in their praise for your technical capabilities and problem-solving mindset.

We would like to formally extend an offer to join our engineering team as {job_title}.

Let us know when you are available for a brief call to walk through the offer details and compensation package.

Warm regards,
Talent Acquisition Team"""
    else:
        return f"""Subject: Update on your application for {job_title}

Dear {candidate_name},

Thank you for taking the time to share your background and interview with us for the {job_title} position.

While your qualifications are impressive, we have decided to move forward with another candidate whose experience more closely aligns with our immediate requirements for this specific role.

We truly appreciate your interest in our team and wish you the very best in your job search.

Sincerely,
Talent Acquisition Team"""


def redact_pii(text: str) -> str:
    """
    Redacts personally identifiable information (Names, Emails, Phones) for bias-free screening.
    """
    redacted = text
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}'

    redacted = re.sub(email_pattern, "[EMAIL REDACTED]", redacted)
    redacted = re.sub(phone_pattern, "[PHONE REDACTED]", redacted)

    if nlp:
        try:
            doc = nlp(redacted[:5000])
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    redacted = redacted.replace(ent.text, "[CANDIDATE NAME]")
        except Exception:
            pass

    return redacted


def correct_bias(text: str) -> List[str]:
    """
    Identifies non-inclusive or biased terminology in job descriptions.
    """
    biased_words = {
        "ninja": "skilled developer / specialist",
        "rockstar": "high-performing engineer",
        "guru": "subject matter expert",
        "he/she": "they",
        "guys": "team / colleagues",
        "young": "energetic / innovative",
        "aggressive": "driven / ambitious"
    }
    suggestions = []
    lower_text = f" {text.lower()} "
    for word, replacement in biased_words.items():
        if f" {word} " in lower_text:
            suggestions.append(f"Consider replacing '{word}' with '{replacement}' for more inclusive recruiting.")
    return suggestions
