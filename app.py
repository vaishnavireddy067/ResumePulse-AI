"""
HireMind AI — Agentic Talent Intelligence Platform
Comprehensive Multi-Agent Recruitment, ATS Screening, Coding Assessments & Talent Analytics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json

from utils.parser import extract_text_from_pdf, extract_text_from_docx
from utils.extractor import extract_contact_info, extract_name
from utils.ai_features import correct_bias
from utils.agentic_pipeline import HireMindOrchestrator
from utils.rag import create_resume_vector_db, query_resume

# --- Page Configuration ---
st.set_page_config(
    page_title="HireMind AI — Agentic Talent Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Modern Sleek Styling ---
st.markdown("""
<style>
    /* Global Base */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        padding: 24px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.3);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 6px;
        background: linear-gradient(90deg, #FFFFFF, #C7D2FE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #E0E7FF;
        font-weight: 400;
    }
    
    /* Card Component */
    .metric-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.08);
    }
    
    /* Status Badges */
    .badge-high {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-med {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-low {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    /* Skill Pill */
    .skill-pill {
        background-color: #EEF2FF;
        color: #3730A3;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 2px 4px 2px 0;
        display: inline-block;
    }
    .skill-pill-missing {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 2px 4px 2px 0;
        display: inline-block;
    }
    
    /* Button Styles */
    .stButton>button {
        background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 20px;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #4338CA 0%, #3730A3 100%);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35);
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = []
if "vector_stores" not in st.session_state:
    st.session_state.vector_stores = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

# --- Header Banner ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🧠 HireMind AI</div>
    <div class="hero-subtitle">Agentic Talent Intelligence Platform • Autonomous Resume Screening, Skill Matching, Coding Assessments & Interview Intelligence</div>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚡ Agent Configuration")
    
    provider_choice = st.selectbox(
        "AI Engine Provider",
        ("Auto (Best Available)", "Google Gemini", "Groq (Llama 3.3 70B)", "Heuristics / Local Mode"),
        help="Select the LLM engine powering the collaborative agents."
    )
    
    provider_map = {
        "Auto (Best Available)": "auto",
        "Google Gemini": "gemini",
        "Groq (Llama 3.3 70B)": "groq",
        "Heuristics / Local Mode": "local"
    }
    selected_provider = provider_map[provider_choice]
    
    st.subheader("🔑 API Credentials")
    gemini_key_input = st.text_input(
        "Gemini API Key",
        value=os.environ.get("GEMINI_API_KEY", ""),
        type="password",
        help="Required if using Google Gemini"
    )
    if gemini_key_input:
        os.environ["GEMINI_API_KEY"] = gemini_key_input

    groq_key_input = st.text_input(
        "Groq API Key",
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
        help="Required if using Groq Llama 3"
    )
    if groq_key_input:
        os.environ["GROQ_API_KEY"] = groq_key_input

    st.divider()
    
    st.subheader("🛡️ Screening & Assessment Settings")
    bias_free_mode = st.toggle(
        "Bias-Free Screening (PII Redaction)",
        value=False,
        help="Anonymizes candidate names, contact details, and demographic markers for fair evaluation."
    )
    
    min_score_filter = st.slider(
        "Minimum ATS Score Threshold (%)",
        min_value=0,
        max_value=100,
        value=30,
        help="Filter out candidates falling below this cutoff score."
    )
    
    seniority_target = st.selectbox(
        "Target Role Seniority",
        ("Junior / Entry", "Mid-Level", "Senior Engineer", "Lead / Principal Architect")
    )
    
    st.divider()
    
    st.markdown("""
    **Active AI Agents:**
    - 🕵️ **Screening & ATS Agent**
    - 🎯 **Skill & Gap Analyzer Agent**
    - 💻 **Coding Assessment Agent**
    - 🎙️ **Interview Rubric Agent**
    - 🛡️ **Authenticity & Risk Agent**
    - 💬 **RAG Resume Chatbot Agent**
    - 📧 **Automated Outreach Agent**
    """)

# --- Top Inputs: Job Description & Resumes ---
col_jd, col_files = st.columns([1.1, 0.9])

with col_jd:
    st.subheader("1️⃣ Job Description (JD)")
    job_description = st.text_area(
        "Paste the Target Role Requirements:",
        height=220,
        placeholder="e.g. Senior Full-Stack Engineer with 4+ years experience in Python, FastAPI, React, Cloud Architecture, and Docker. Experience with agentic AI and LLM workflows is a plus..."
    )
    
    if job_description:
        bias_suggestions = correct_bias(job_description)
        if bias_suggestions:
            with st.expander("⚠️ Inclusive Language Suggestions"):
                for s in bias_suggestions:
                    st.info(s)
        elif len(job_description) > 60:
            st.success("✅ Job Description language is inclusive and clear!")

with col_files:
    st.subheader("2️⃣ Candidate Resumes")
    uploaded_files = st.file_uploader(
        "Upload Resumes (PDF / DOCX):",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Upload single or batch candidate resumes."
    )
    
    launch_btn = st.button("🚀 Run HireMind Multi-Agent Screening", use_container_width=True)

# --- Pipeline Execution ---
if launch_btn:
    if not job_description.strip():
        st.error("⚠️ Please provide a Job Description before running screening.")
    elif not uploaded_files:
        st.error("⚠️ Please upload at least one candidate resume.")
    else:
        with st.spinner("🤖 Multi-Agent Pipeline executing across candidates..."):
            orchestrator = HireMindOrchestrator(provider=selected_provider)
            results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                filename = uploaded_file.name
                status_text.text(f"Agent Pipeline processing ({idx+1}/{len(uploaded_files)}): {filename}")
                
                # Extract text
                if filename.lower().endswith(".pdf"):
                    resume_text = extract_text_from_pdf(uploaded_file)
                elif filename.lower().endswith(".docx"):
                    resume_text = extract_text_from_docx(uploaded_file)
                else:
                    resume_text = ""
                
                if resume_text.strip():
                    # Run Orchestrated Agents
                    candidate_eval = orchestrator.process_candidate(
                        jd=job_description,
                        resume_text=resume_text,
                        filename=filename,
                        bias_free=bias_free_mode,
                        run_deep_agents=True
                    )
                    
                    # Store vector store for RAG
                    cand_id = candidate_eval["name"]
                    st.session_state.vector_stores[cand_id] = create_resume_vector_db(resume_text, cand_id)
                    results.append(candidate_eval)
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
                
            status_text.empty()
            progress_bar.empty()
            
            # Sort by ATS Score descending
            results.sort(key=lambda x: x["ats_score"], reverse=True)
            st.session_state.evaluation_results = results
            st.success(f"✅ Evaluation complete! Processed {len(results)} candidates.")

# --- Results Presentation ---
if st.session_state.evaluation_results:
    results = [c for c in st.session_state.evaluation_results if c["ats_score"] >= min_score_filter]
    
    if not results:
        st.warning(f"No candidates met the minimum score cutoff of {min_score_filter}%. Try lowering the threshold.")
    else:
        # High-level Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        avg_score = round(sum(c["ats_score"] for c in results) / len(results), 1)
        top_cand = results[0]["name"]
        high_prob_count = sum(1 for c in results if c["hiring_prediction"]["tier"] == "High")
        
        with m1:
            st.metric("Total Candidates Evaluated", f"{len(results)}")
        with m2:
            st.metric("Average ATS Score", f"{avg_score}%")
        with m3:
            st.metric("Top Ranked Match", f"{results[0]['ats_score']}%", f"{top_cand}")
        with m4:
            st.metric("High-Probability Matches", f"{high_prob_count}", f"{(high_prob_count/len(results)*100):.0f}% of pool")

        st.markdown("---")
        
        # Tabs for Comprehensive Intelligence
        t_rankings, t_deepdive, t_coding, t_interview, t_rag, t_analytics, t_outreach = st.tabs([
            "🏆 1. ATS Leaderboard",
            "🎯 2. Candidate Deep-Dive",
            "💻 3. Coding Assessment Studio",
            "🎙️ 4. Interview Rubrics",
            "💬 5. HireMind RAG Chatbot",
            "📊 6. Talent Analytics",
            "📧 7. Automated Outreach"
        ])

        # === TAB 1: ATS LEADERBOARD ===
        with t_rankings:
            st.subheader("Candidate Rankings & Screening Overview")
            
            table_data = []
            for rank, c in enumerate(results, start=1):
                table_data.append({
                    "Rank": rank,
                    "Candidate Name": c["name"],
                    "Filename": c["filename"],
                    "ATS Score (%)": c["ats_score"],
                    "Semantic Match": f"{c['semantic_score']}%",
                    "Hiring Probability": c["hiring_prediction"]["label"],
                    "Authenticity": c["authenticity"]["status"],
                    "Matched Skills": len(c["matched_keywords"]),
                    "Email": c["email"],
                    "Phone": c["phone"]
                })
            
            df_table = pd.DataFrame(table_data)
            st.dataframe(
                df_table,
                use_container_width=True,
                column_config={
                    "ATS Score (%)": st.column_config.ProgressColumn(
                        "ATS Score (%)",
                        help="Blended score from semantic AI and keyword density",
                        format="%d%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "Rank": st.column_config.NumberColumn("Rank", width="small")
                },
                hide_index=True
            )
            
            # Export to CSV
            csv_export = df_table.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export Talent Intelligence Report (CSV)",
                data=csv_export,
                file_name="hiremind_talent_report.csv",
                mime="text/csv"
            )

        # === TAB 2: CANDIDATE DEEP-DIVE ===
        with t_deepdive:
            candidate_names = [c["name"] for c in results]
            selected_cand_name = st.selectbox("Select Candidate for In-Depth Profile Analysis:", candidate_names, key="deepdive_select")
            cand = next((c for c in results if c["name"] == selected_cand_name), results[0])
            
            c_left, c_right = st.columns([1, 1.2])
            
            with c_left:
                st.markdown(f"### 👤 {cand['name']}")
                st.markdown(f"**Source File:** `{cand['filename']}`")
                st.markdown(f"**Email:** `{cand['email']}` | **Phone:** `{cand['phone']}`")
                
                # Hiring prediction badge
                tier_color = cand["hiring_prediction"]["color"]
                st.markdown(f"""
                <div style="background-color: {tier_color}20; border-left: 4px solid {tier_color}; padding: 12px; border-radius: 8px; margin: 12px 0;">
                    <div style="font-weight: 700; color: {tier_color}; font-size: 1.1rem;">{cand['hiring_prediction']['badge']}</div>
                    <div style="color: #374151; font-size: 0.95rem; margin-top: 4px;"><b>Recommendation:</b> {cand['hiring_prediction']['recommendation']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Authenticity Audit
                auth = cand["authenticity"]
                auth_color = "#10B981" if auth["status"] == "Passed" else "#F59E0B" if auth["status"] == "Review Needed" else "#EF4444"
                with st.expander(f"🛡️ Authenticity & Red-Flag Audit ({auth['status']})", expanded=True):
                    for flag in auth["flags"]:
                        st.markdown(f"- {flag}")
            
            with c_right:
                st.markdown("#### 🎯 Skill Competency & Gap Analysis")
                if "skill_analysis" in cand:
                    sk = cand["skill_analysis"]
                    
                    st.markdown("**✅ Verified Matching Skills:**")
                    matched_html = "".join([f"<span class='skill-pill'>{skill}</span>" for skill in sk.get("matched_skills", cand["matched_keywords"][:8])])
                    st.markdown(matched_html, unsafe_allow_html=True)
                    
                    st.markdown("<br>**❌ Missing Critical Skills:**", unsafe_allow_html=True)
                    missing_html = "".join([f"<span class='skill-pill-missing'>{skill}</span>" for skill in sk.get("missing_critical_skills", cand["missing_keywords"][:5])])
                    st.markdown(missing_html, unsafe_allow_html=True)
                    
                    st.markdown(f"<br>**💡 Strengths Summary:** {sk.get('strengths_summary', 'Demonstrates solid alignment with core responsibilities.')}", unsafe_allow_html=True)
                    st.markdown(f"**📈 Growth / Ramp-Up Areas:** {sk.get('growth_areas', 'May require domain onboarding for specialized tools.')}", unsafe_allow_html=True)

        # === TAB 3: CODING ASSESSMENT STUDIO ===
        with t_coding:
            st.subheader("💻 Candidate-Tailored Coding Assessment")
            candidate_names = [c["name"] for c in results]
            coding_cand_name = st.selectbox("Select Candidate:", candidate_names, key="coding_select")
            cand = next((c for c in results if c["name"] == coding_cand_name), results[0])
            
            if "coding_challenge" in cand:
                challenge = cand["coding_challenge"]
                
                c1, c2, c3 = st.columns([2, 1, 1])
                with c1:
                    st.markdown(f"### ⚡ {challenge.get('title', 'Live Coding Challenge')}")
                with c2:
                    st.markdown(f"**Difficulty:** `{challenge.get('difficulty', seniority_target)}`")
                with c3:
                    st.markdown(f"**Estimated Time:** `{challenge.get('estimated_time', '45 mins')}`")
                
                # Tech tags
                tech_tags = "".join([f"<span class='skill-pill'>{t}</span>" for t in challenge.get("core_technologies", ["Python"])])
                st.markdown(tech_tags, unsafe_allow_html=True)
                
                st.markdown("#### 📝 Problem Statement")
                st.markdown(challenge.get("problem_statement", "Implement solution."))
                
                st.markdown("#### 💻 Starter Boilerplate Code")
                st.code(challenge.get("boilerplate_code", "# Write solution here"), language="python")
                
                # Test Cases
                st.markdown("#### 🧪 Sample Test Cases")
                test_cases = challenge.get("sample_test_cases", [])
                if test_cases:
                    df_tests = pd.DataFrame(test_cases)
                    st.table(df_tests)
                
                # Rubric
                st.markdown("#### 📐 Evaluation Criteria & Weightage")
                rubric_items = challenge.get("evaluation_rubric", [])
                if rubric_items:
                    for r in rubric_items:
                        st.markdown(f"- **{r.get('criterion', 'Quality')}** (`{r.get('weight', '')}`): {r.get('description', '')}")

        # === TAB 4: INTERVIEW RUBRICS ===
        with t_interview:
            st.subheader("🎙️ Structured Interview Questions & Scoring Rubrics")
            candidate_names = [c["name"] for c in results]
            interview_cand_name = st.selectbox("Select Candidate:", candidate_names, key="interview_select")
            cand = next((c for c in results if c["name"] == interview_cand_name), results[0])
            
            if "interview_rubric" in cand:
                rubric = cand["interview_rubric"]
                
                st.markdown("#### 🔬 Deep Technical Questions")
                for q_idx, tq in enumerate(rubric.get("technical_questions", []), 1):
                    with st.expander(f"Question {q_idx}: {tq.get('question', '')}", expanded=True):
                        st.markdown(f"🎯 **Target Competency:** `{tq.get('target_competency', 'Technical')}`")
                        st.markdown(f"🟢 **Good Answer Signals:** {tq.get('good_answer_signals', '')}")
                        st.markdown(f"🔴 **Red Flag Signals:** {tq.get('red_flag_signals', '')}")
                
                st.markdown("#### 🤝 Behavioral & Leadership Questions")
                for b_idx, bq in enumerate(rubric.get("behavioral_questions", []), 1):
                    with st.expander(f"Behavioral Scenario {b_idx}: {bq.get('question', '')}"):
                        st.markdown(f"🎯 **Target Competency:** `{bq.get('target_competency', 'Leadership')}`")
                        st.markdown(f"🟢 **Good Answer Signals:** {bq.get('good_answer_signals', '')}")
                        st.markdown(f"🔴 **Red Flag Signals:** {bq.get('red_flag_signals', '')}")
                
                arch = rubric.get("architecture_scenario", {})
                if arch:
                    st.markdown("#### 🏛️ Real-World Architectural Challenge")
                    st.info(f"**Scenario:** {arch.get('scenario', '')}")
                    st.markdown("**Key Evaluation Touchpoints:**")
                    for pt in arch.get("evaluation_points", []):
                        st.markdown(f"- {pt}")

        # === TAB 5: RAG CHATBOT ===
        with t_rag:
            st.subheader("💬 HireMind RAG Resume Chatbot")
            candidate_names = [c["name"] for c in results]
            rag_cand_name = st.selectbox("Select Candidate to Chat With:", candidate_names, key="rag_select")
            cand = next((c for c in results if c["name"] == rag_cand_name), results[0])
            
            st.info(f"💡 You are currently querying **{cand['name']}'s** resume. Ask specific questions about projects, timeline, skills, or leadership experience.")
            
            # User query
            sample_questions = [
                "Summarize their most significant career achievements.",
                "What specific experience do they have with the required tech stack?",
                "What are potential weaknesses or gaps for this role?",
                "Did they lead any teams or mentor junior engineers?"
            ]
            
            chosen_sample = st.selectbox("Or choose a pre-configured prompt:", ["-- Choose a quick question --"] + sample_questions)
            
            user_question = st.text_input(
                "Ask anything about this resume:",
                value=chosen_sample if chosen_sample != "-- Choose a quick question --" else "",
                placeholder="e.g. Does this candidate have production experience with Kubernetes and CI/CD?"
            )
            
            if st.button("Ask HireMind AI", key="rag_ask_btn"):
                if user_question.strip():
                    with st.spinner(f"Querying {cand['name']}'s resume with AI..."):
                        v_store = st.session_state.vector_stores.get(cand["name"])
                        response = query_resume(v_store, user_question, candidate_name=cand["name"], provider=selected_provider)
                        st.markdown("### 🤖 Response")
                        st.markdown(response)

        # === TAB 6: TALENT ANALYTICS ===
        with t_analytics:
            st.subheader("📊 Comparative Talent Pool Analytics")
            
            g1, g2 = st.columns(2)
            
            with g1:
                # Bar Chart of ATS Scores
                df_scores = pd.DataFrame({
                    "Candidate": [c["name"] for c in results],
                    "ATS Score": [c["ats_score"] for c in results],
                    "Hiring Tier": [c["hiring_prediction"]["tier"] for c in results]
                })
                fig_bar = px.bar(
                    df_scores,
                    x="Candidate",
                    y="ATS Score",
                    color="Hiring Tier",
                    color_discrete_map={"High": "#10B981", "Medium": "#F59E0B", "Low": "#EF4444"},
                    title="Candidate ATS Match Score Comparison (%)"
                )
                fig_bar.update_layout(xaxis_tickangle=-45, height=380)
                st.plotly_chart(fig_bar, use_container_width=True)
                
            with g2:
                # Donut Chart for Hiring Probability distribution
                tier_counts = df_scores["Hiring Tier"].value_counts().reset_index()
                tier_counts.columns = ["Tier", "Count"]
                fig_donut = px.pie(
                    tier_counts,
                    names="Tier",
                    values="Count",
                    hole=0.45,
                    color="Tier",
                    color_discrete_map={"High": "#10B981", "Medium": "#F59E0B", "Low": "#EF4444"},
                    title="Talent Pool Fit Distribution"
                )
                fig_donut.update_layout(height=380)
                st.plotly_chart(fig_donut, use_container_width=True)

        # === TAB 7: AUTOMATED OUTREACH ===
        with t_outreach:
            st.subheader("📧 Automated Recruiter Outreach & Communications")
            candidate_names = [c["name"] for c in results]
            outreach_cand_name = st.selectbox("Select Candidate:", candidate_names, key="outreach_select")
            cand = next((c for c in results if c["name"] == outreach_cand_name), results[0])
            
            col_mail_type, col_mail_pos = st.columns(2)
            with col_mail_type:
                email_type = st.selectbox(
                    "Email Template Type:",
                    ("interview", "offer", "rejection"),
                    format_func=lambda x: "🎙️ Interview Invitation" if x == "interview" else "🎉 Formal Offer Discussion" if x == "offer" else "✉️ Polite Rejection"
                )
            with col_mail_pos:
                job_title_input = st.text_input("Role Title for Email:", value="Senior Software Engineer")
                
            custom_note = st.text_area("Additional Custom Note (Optional):", placeholder="e.g. Highlighted strong performance on system design...")
            
            if st.button("✨ Generate Personalized Email Draft", use_container_width=True):
                with st.spinner("Drafting email..."):
                    orchestrator = HireMindOrchestrator(provider=selected_provider)
                    email_draft = orchestrator.outreach_agent.run(
                        candidate_name=cand["name"],
                        job_title=job_title_input,
                        status=email_type,
                        feedback=custom_note,
                        provider=selected_provider
                    )
                    st.text_area("Ready-to-Send Email Copy:", value=email_draft, height=260)
                    st.success("Draft ready! Copy and send via your ATS or email client.")
