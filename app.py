import streamlit as st
import pandas as pd
import os
from utils.parser import extract_text_from_pdf, extract_text_from_docx
from utils.matcher import rank_resumes
from utils.extractor import extract_contact_info, extract_name
from utils.ai_features import semantic_match, redact_pii, correct_bias, generate_interview_questions, predict_hiring_chance, check_authenticity, generate_email

from utils.translator import translate_resume
from utils.rag import create_resume_vector_db, query_resume

# Page Configuration
st.set_page_config(
    page_title="ResumePulse AI - Smart ATS Screening",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for a "Premium" look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    h1 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stFileUploader {
        border: 2px dashed #4CAF50;
        padding: 20px;
        border-radius: 10px;
    }
    .stExpander {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Application Header
st.title("🤖 ResumePulse AI - Smart ATS Screener")
st.markdown("### Intelligent Candidate Shortlisting & Ranking System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("Upload resumes and a job description to get started.")
    
    ranking_method = st.selectbox(
        "Ranking Algorithm",
        ("TF-IDF Vectorization (Fast)", "Cosine Similarity (Base)", "AI Semantic (Groq Llama 3) - Accurate")
    )
    
    st.divider()
    
    # -- Features Toggles --
    st.subheader("🔍 Screening Mode")
    bias_free_mode = st.toggle("Bias-Free Mode (Hide PII)", help="Hides Name, Email, and Phone to reduce unconscious bias.")
    
    min_score = st.slider(
        "Minimum Match Score (%)", 
        min_value=0, 
        max_value=100, 
        value=0,
        help="Filter out candidates below this score."
    )

# Main Content Layout
col1, col2 = st.columns([1, 1])

# Left Column: Inputs
with col1:
    st.subheader("Job Description")
    job_description = st.text_area(
        "Paste the Job Description (JD) here:",
        height=250,
        placeholder="e.g. Seeking a Python Developer with experience in Django and Machine Learning..."
    )
    
    # -- JD Bias Checker --
    if job_description:
        suggestions = correct_bias(job_description)
        if suggestions:
            with st.expander("⚠️ JD Improvement Suggestions (Bias Check)"):
                for s in suggestions:
                    st.warning(s)
        else:
             if len(job_description) > 50:
                 st.success("✅ JD looks neutral and inclusive!")

    st.subheader("Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload Candidate Resumes (PDF/DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

# Right Column: Results
with col2:
    st.subheader("Ranking & Analysis")
    
    submit_button = st.button("🚀 Analyze & Rank Candidates")
    
    if submit_button:
        if not job_description:
            st.error("⚠️ Please enter a Job Description first.")
        elif not uploaded_files:
            st.error("⚠️ Please upload at least one resume.")
        else:
            with st.spinner("Reading resumes, translating (if needed), and matching..."):
                results = []
                # Dictionary to store vector stores for chatting
                st.session_state['vector_stores'] = {} 
                
                for file in uploaded_files:
                    text = ""
                    if file.name.endswith(".pdf"):
                        text = extract_text_from_pdf(file)
                    elif file.name.endswith(".docx"):
                        text = extract_text_from_docx(file)
                    
                    if text:
                        # 0. Translate if multi-language
                        text = translate_resume(text)
                        
                        # 1. Rank
                        if "Groq" in ranking_method or "BERT" in ranking_method:
                            score = semantic_match(job_description, text)
                            _, missing_keywords, common_keywords = rank_resumes(job_description, text) 
                        else:
                            score, missing_keywords, common_keywords = rank_resumes(job_description, text)
                        
                        # 2. Extract Entities
                        contact_info = extract_contact_info(text)
                        
                        if bias_free_mode:
                            candidate_name = f"Candidate {len(results)+1}"
                        else:
                            candidate_name = extract_name(text)
                        
                        # 3. Handle Bias-Free Mode
                        if bias_free_mode:
                            contact_info['email'] = "HIDDEN"
                            contact_info['phone'] = "HIDDEN"
                            preview_text = redact_pii(text[:3000])
                        else:
                            preview_text = text[:3000]
                        
                        # 4. Generate Interview Questions
                        interview_questions = generate_interview_questions(text)

                        # 5. Create Vector DB for Chat (Store in session state mapped to ID)
                        # We use index as ID for simplicity
                        candidate_id = len(results)
                        st.session_state['vector_stores'][candidate_id] = create_resume_vector_db(text)
                        
                        # [NEW] AI Predictions
                        hiring_chance, color = predict_hiring_chance(score)
                        auth_flags = check_authenticity(text)
                        
                        results.append({
                            "ID": candidate_id,
                            "Candidate Name": candidate_name,
                            "Email": contact_info['email'],
                            "Phone": contact_info['phone'],
                            "Match Score": score,
                            "Hiring Chance": hiring_chance,
                            "Authenticity": auth_flags,
                            "Missing Keywords": ", ".join(missing_keywords[:5]) + ("..." if len(missing_keywords) > 5 else "") if missing_keywords else "None",
                            "Common Keywords": common_keywords,
                            "Full Missing Keywords": missing_keywords,
                            "Text Preview": preview_text + "...",
                            "Interview Questions": interview_questions
                        })
                
                if results:
                    df = pd.DataFrame(results)
                    df.sort_values(by="Match Score", ascending=False, inplace=True)
                    st.session_state['analysis_results'] = df
                    st.session_state['analysis_done'] = True
                    st.toast("Analysis Complete!", icon="✅")
                else:
                    st.error("Could not extract text from uploaded files.")

    # -- Display Results (Persist across reruns) --
    if st.session_state.get('analysis_done') and 'analysis_results' in st.session_state:
        df = st.session_state['analysis_results']
        
        # -- Apply Filter (Dynamic) --
        original_count = len(df)
        filtered_df = df[df["Match Score"] >= min_score]
        filtered_count = len(filtered_df)
        
        st.success(f"✅ Analysis Complete! Showing {filtered_count} of {original_count} candidates.")
        
        if not filtered_df.empty:
            # -- Display Rank Table --
            st.markdown("### 🏆 Top Candidates")
            
            cols_to_show = ["Candidate Name", "Match Score", "Missing Keywords"]
            if not bias_free_mode:
                cols_to_show = ["Candidate Name", "Match Score", "Email", "Phone", "Missing Keywords"]
                
            st.dataframe(
                filtered_df[cols_to_show],
                column_config={
                    "Match Score": st.column_config.ProgressColumn(
                        "Match Score",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                },
                use_container_width=True
            )

            # -- Visualizations --
            st.markdown("### 📊 Analysis Dashboard")
            import plotly.express as px
            fig = px.bar(filtered_df, x="Candidate Name", y="Match Score", 
                            title="Candidate Ranking", 
                            color="Match Score", 
                            color_continuous_scale="Viridis")
            st.plotly_chart(fig, use_container_width=True)

            # -- Explainable AI Section --
            st.markdown("### 🧠 Deep Dive, Chat & Compare")
            for index, row in filtered_df.iterrows():
                # Extract simplified role from JD (first 5 words) or default
                role = " ".join(job_description.split()[:5]) + "..." if job_description else "Open Role"
                
                with st.expander(f"🔎 Inspector: {row['Candidate Name']} (Score: {row['Match Score']}%)"):
                    
                    st.caption(f"**ID:** {row['ID']} | **Email:** {row['Email']}")
                    
                    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🧠 Insights", "🤖 Chat with Resume", "🎤 Interview Qs", "📄 Text", "📧 Outreach"])
                    
                    with tab1:
                        # Hiring Prediction Badge
                        chance = row.get('Hiring Chance', 'Unknown')
                        color = "green" if "High" in chance else "orange" if "Medium" in chance else "red"
                        
                        st.markdown(f"### 🔮 Hiring Prediction: :{color}[{chance}]")
                        
                        # Authenticity Check
                        auth = row.get('Authenticity', [])
                        if "✅ Authentic" in auth:
                            st.success("✅ Resume Authenticity Verified")
                        else:
                            st.error(f"⚠️ Authenticity Flags: {', '.join(auth)}")
                            
                        st.divider()
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("#### ✅ Strengths")
                            if row['Common Keywords']:
                                tags = " ".join([f"`{kw}`" for kw in row['Common Keywords']])
                                st.markdown(tags)
                            else:
                                st.info("No direct keyword matches.")
                        
                        with c2:
                            st.markdown("#### ⚠️ Missing Skills")
                            if row['Full Missing Keywords']:
                                tags = " ".join([f"`{kw}`" for kw in row['Full Missing Keywords']])
                                st.markdown(tags)
                            else:
                                st.success("All matches found!")
                    
                    with tab5:
                        st.markdown("#### 📧 Automated Email Draft")
                        subject, body = generate_email(row['Candidate Name'], role, row['Match Score'])
                        st.text_input("Subject", value=subject)
                        st.text_area("Body", value=body, height=200)
                        st.button(f"Copy Email for {row['Candidate Name']}", key=f"email_{row['ID']}")
                    
                    with tab2:
                        st.markdown("#### 💬 Ask questions about this specific candidate")
                        user_q = st.text_input(f"Ask about {row['Candidate Name']}:", key=f"q_{row['ID']}")
                        if user_q:
                            if row['ID'] in st.session_state['vector_stores']:
                                answer = query_resume(st.session_state['vector_stores'][row['ID']], user_q)
                                st.markdown(answer)
                            else:
                                st.error("Chat index not found. Please re-analyze.")

                    with tab3:
                        st.markdown("#### 🤖 Suggested Interview Questions")
                        for q in row['Interview Questions']:
                            st.write(f"- {q}")
                    
                    with tab4:
                        st.text_area("Content", row.get("Text Preview", "No text"), height=300, key=f"preview_{index}")

            # -- Multi-Resume Comparison --
            st.markdown("### ⚔️ Compare Candidates")
            candidates_to_compare = st.multiselect(
                "Select 2 or more candidates to compare side-by-side:",
                filtered_df["Candidate Name"].tolist()
            )
            
            if len(candidates_to_compare) > 1:
                comp_df = filtered_df[filtered_df["Candidate Name"].isin(candidates_to_compare)]
                
                # Transpose for side-by-side view
                comparison_data = {
                    "Metric": ["Match Score", "Email", "Phone", "Strengths", "Missing Skills"],
                }
                
                for i, cand in comp_df.iterrows():
                    strengths = ", ".join(list(cand['Common Keywords'])[:5]) if cand['Common Keywords'] else "None"
                    missing = ", ".join(list(cand['Full Missing Keywords'])[:5]) if cand['Full Missing Keywords'] else "None"
                    
                    comparison_data[cand['Candidate Name']] = [
                        f"{cand['Match Score']}%",
                        cand['Email'],
                        cand['Phone'],
                        strengths,
                        missing
                    ]
                    
                st.table(pd.DataFrame(comparison_data))
                
        else:
            st.warning(f"No candidates met the minimum score of {min_score}%.")
