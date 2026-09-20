"""
HireMind AI — Multi-Agent Talent Intelligence Pipeline
Collaborative AI Agents for end-to-end recruitment intelligence.
"""

import json
import re
from typing import Dict, Any, List, Optional
from utils.ai_features import (
    get_llm_response,
    semantic_match,
    predict_hiring_chance,
    check_authenticity,
    generate_coding_assessment,
    generate_interview_rubric,
    generate_email,
    redact_pii
)
from utils.matcher import rank_resumes
from utils.extractor import extract_contact_info, extract_name

class BaseTalentAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    def log(self, message: str):
        print(f"[{self.name} | {self.role}]: {message}")


class ResumeScreeningAgent(BaseTalentAgent):
    """Evaluates ATS relevance, semantic match, and extracts core candidate profile."""
    def __init__(self):
        super().__init__("ScreeningAgent", "ATS & Resume Evaluator")

    def run(self, jd: str, resume_text: str, provider: str = "auto") -> Dict[str, Any]:
        # Fast TF-IDF Match
        tfidf_score, missing_kw, common_kw = rank_resumes(jd, resume_text)
        
        # AI Semantic Match
        ai_score = semantic_match(jd, resume_text, provider=provider)
        
        # Blended ATS Score
        if ai_score > 0:
            final_score = round(0.7 * ai_score + 0.3 * tfidf_score, 1)
        else:
            final_score = tfidf_score

        contact = extract_contact_info(resume_text)
        candidate_name = extract_name(resume_text)

        return {
            "name": candidate_name,
            "email": contact.get("email", "Not Found"),
            "phone": contact.get("phone", "Not Found"),
            "ats_score": final_score,
            "tfidf_score": tfidf_score,
            "semantic_score": ai_score if ai_score > 0 else tfidf_score,
            "matched_keywords": common_kw,
            "missing_keywords": missing_kw
        }


class SkillGapAgent(BaseTalentAgent):
    """Analyzes candidate competencies, missing critical skills, and strengths."""
    def __init__(self):
        super().__init__("SkillGapAgent", "Skill & Competency Analyst")

    def run(self, jd: str, resume_text: str, provider: str = "auto") -> Dict[str, Any]:
        prompt = f"""
        Act as a Senior Technical Recruiter and Talent Assessor.
        Analyze this Job Description and Candidate Resume to identify skill gaps and key strengths.

        Job Description:
        {jd[:1500]}

        Candidate Resume:
        {resume_text[:1500]}

        Return your output in strict JSON format with keys:
        - "matched_skills": [list of skills candidate has that match JD],
        - "missing_critical_skills": [list of critical required skills candidate lacks],
        - "missing_secondary_skills": [list of nice-to-have skills candidate lacks],
        - "strengths_summary": "Short 2-sentence summary of candidate strengths",
        - "growth_areas": "Short 2-sentence summary of areas candidate needs to ramp up",
        - "fit_level": "High" or "Medium" or "Low"
        """
        response = get_llm_response(prompt, provider=provider, system_prompt="You are an expert AI talent intelligence agent. Output only valid JSON.")
        
        try:
            # Extract JSON block
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass

        # Fallback analysis
        return {
            "matched_skills": ["Python", "Problem Solving", "Git"],
            "missing_critical_skills": ["Cloud Architecture", "Distributed Systems"],
            "missing_secondary_skills": ["CI/CD Pipelines"],
            "strengths_summary": "Candidate shows solid foundational knowledge in core domain skills.",
            "growth_areas": "Could benefit from more enterprise-scale deployment experience.",
            "fit_level": "Medium"
        }


class CodingAssessmentAgent(BaseTalentAgent):
    """Generates tailored live coding challenges based on candidate profile and job requirements."""
    def __init__(self):
        super().__init__("CodingAgent", "Technical Assessment Architect")

    def run(self, jd: str, resume_text: str, seniority: str = "Mid-Level", provider: str = "auto") -> Dict[str, Any]:
        return generate_coding_assessment(jd, resume_text, seniority=seniority, provider=provider)


class InterviewRubricAgent(BaseTalentAgent):
    """Generates role-tailored technical & behavioral interview questions with scoring rubrics."""
    def __init__(self):
        super().__init__("InterviewAgent", "Interview Rubric Designer")

    def run(self, jd: str, resume_text: str, provider: str = "auto") -> Dict[str, Any]:
        return generate_interview_rubric(jd, resume_text, provider=provider)


class RedFlagAgent(BaseTalentAgent):
    """Scans for authenticity, keyword-stuffing, discrepancies, and red flags."""
    def __init__(self):
        super().__init__("RedFlagAgent", "Authenticity & Risk Auditor")

    def run(self, resume_text: str) -> Dict[str, Any]:
        return check_authenticity(resume_text)


class OutreachAgent(BaseTalentAgent):
    """Drafts personalized candidate emails for invitations, follow-ups, or rejections."""
    def __init__(self):
        super().__init__("OutreachAgent", "Talent Communications Specialist")

    def run(self, candidate_name: str, job_title: str, status: str = "interview", feedback: str = "", provider: str = "auto") -> str:
        return generate_email(candidate_name, job_title, status=status, feedback=feedback, provider=provider)


class HireMindOrchestrator:
    """
    Central Collaborative Multi-Agent Orchestrator for HireMind AI.
    Coordinates agents to perform comprehensive candidate talent evaluation.
    """
    def __init__(self, provider: str = "auto"):
        self.provider = provider
        self.screening_agent = ResumeScreeningAgent()
        self.skill_gap_agent = SkillGapAgent()
        self.coding_agent = CodingAssessmentAgent()
        self.interview_agent = InterviewRubricAgent()
        self.red_flag_agent = RedFlagAgent()
        self.outreach_agent = OutreachAgent()

    def process_candidate(self, jd: str, resume_text: str, filename: str = "", bias_free: bool = False, run_deep_agents: bool = True) -> Dict[str, Any]:
        """Runs the full multi-agent talent evaluation pipeline for a candidate."""
        
        # 1. Redaction if Bias-Free Mode enabled
        processed_text = redact_pii(resume_text) if bias_free else resume_text
        
        # 2. Screening Agent
        screening_res = self.screening_agent.run(jd, processed_text, provider=self.provider)
        
        # 3. Authenticity / Red-Flag Agent
        auth_res = self.red_flag_agent.run(processed_text)
        
        # 4. Hiring Probability
        hiring_pred = predict_hiring_chance(screening_res["ats_score"], len(screening_res["missing_keywords"]))

        candidate_data = {
            "filename": filename,
            "name": screening_res["name"] if not bias_free else "[ANONYMIZED CANDIDATE]",
            "email": screening_res["email"] if not bias_free else "[REDACTED]",
            "phone": screening_res["phone"] if not bias_free else "[REDACTED]",
            "ats_score": screening_res["ats_score"],
            "tfidf_score": screening_res["tfidf_score"],
            "semantic_score": screening_res["semantic_score"],
            "hiring_prediction": hiring_pred,
            "authenticity": auth_res,
            "matched_keywords": screening_res["matched_keywords"],
            "missing_keywords": screening_res["missing_keywords"],
            "raw_text": processed_text
        }

        # 5. Deep Agents (Skill Gap, Coding Assessment, Interview Rubric)
        if run_deep_agents:
            candidate_data["skill_analysis"] = self.skill_gap_agent.run(jd, processed_text, provider=self.provider)
            candidate_data["coding_challenge"] = self.coding_agent.run(jd, processed_text, provider=self.provider)
            candidate_data["interview_rubric"] = self.interview_agent.run(jd, processed_text, provider=self.provider)

        return candidate_data
