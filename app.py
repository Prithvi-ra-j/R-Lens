# ---------- app.py ----------
import streamlit as st
import pandas as pd
import json
import os
from llm_util import (
    build_prompt,
    analyze_with_together,
    extract_llm_output,
    extract_text_from_pdf,
    extract_text_from_docx,
)

st.set_page_config(page_title="LLM Resume Matcher", layout="wide")
st.title("📊 Resume Matcher Dashboard (LLM-powered)")

# --------------------------
# 📤 Upload Files
# --------------------------
st.sidebar.header("📤 Upload Files")
resume_files = st.sidebar.file_uploader(
    "Upload Resume Files", type=["pdf", "docx", "txt"], accept_multiple_files=True
)
# --------------------------
# 📤 JD Input Options
# --------------------------
st.sidebar.header("📥 Job Description Input")

jd_input_mode = st.sidebar.radio(
    "Select how to input Job Descriptions:",
    ("Upload JD Files", "Paste JD Text", "Paste JD URL")
)

jds = {}

if jd_input_mode == "Upload JD Files":
    jd_files = st.sidebar.file_uploader(
        "Upload JD .txt files", type=["txt"], accept_multiple_files=True, key="jd_files"
    )
    for file in jd_files:
        jd_text = file.read().decode("utf-8")
        jds[file.name] = jd_text

elif jd_input_mode == "Paste JD Text":
    jd_text_input = st.sidebar.text_area("Paste Job Description here:", height=300)
    if jd_text_input.strip():
        jds["pasted_jd.txt"] = jd_text_input.strip()

elif jd_input_mode == "Paste JD URL":
    jd_url = st.sidebar.text_input("Paste JD URL (e.g., LinkedIn, Indeed)")
    if jd_url.strip():
        import requests
        try:
            response = requests.get(jd_url)
            if response.status_code == 200:
                jds["jd_from_url.txt"] = response.text
                st.sidebar.success("✅ Fetched JD from URL successfully.")
            else:
                st.sidebar.error(f"❌ Failed to fetch JD. Status code: {response.status_code}")
        except Exception as e:
            st.sidebar.error(f"❌ Error fetching JD: {e}")
# --------------------------
# 🔍 Run Matching
# --------------------------
if st.sidebar.button("🔍 Match Resumes"):
    if not resume_files or not jds:
        st.warning("⚠️ Please upload resumes and provide at least one job description.")
    else:
        results = []
        with st.spinner("⚙️ Evaluating resumes against job descriptions..."):
            for jd_name, jd_text in jds.items():
                for resume in resume_files:
                    ext = resume.name.split(".")[-1].lower()
                    if ext == "pdf":
                        resume_text = extract_text_from_pdf(resume)
                    elif ext == "docx":
                        resume_text = extract_text_from_docx(resume)
                    else:
                        resume_text = resume.read().decode("utf-8")

                    prompt = build_prompt(jd_text, resume_text)
                    raw_output = analyze_with_together(prompt)
                    parsed = extract_llm_output(raw_output, resume.name, jd_name)
                    results.append(parsed)

        # Save results
        with open("resume_match_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        st.success("✅ Matching complete and saved to resume_match_results.json!")

# --------------------------
# 📊 Display Results
# --------------------------
if os.path.exists("resume_match_results.json"):
    with open("resume_match_results.json", "r", encoding="utf-8") as f:
        results = json.load(f)

    if results:
        st.subheader("📝 Raw Evaluation Report")
        options = [f"{row['Resume']} ➝ {row['Job Description']}" for row in results]
        selected_row = st.selectbox("Select a resume + JD pair", options)
        selected_data = next(
            (row for row in results if f"{row['Resume']} ➝ {row['Job Description']}" == selected_row),
            None
        )
        if selected_data:
            st.markdown(f"### 📄 **{selected_data['Resume']}** vs **{selected_data['Job Description']}**")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("🎯 Match Score", f"{selected_data.get('Match Score', 0)} / 100")
            with col2:
                st.metric("🎓 Resume Score", f"{selected_data.get('Resume Score', 0)} / 10")

            st.markdown(f"**📝 Summary:**\n\n{selected_data.get('Summary', 'Not available')}")
            st.markdown("**✅ Strengths:**")
            st.markdown(selected_data.get("Strengths", "Not provided"))

            st.markdown("**🚨 Weaknesses:**")
            st.markdown(selected_data.get("Weaknesses", "Not provided"))

            st.markdown("**📢 Verdict:**")
            st.markdown(selected_data.get("Verdict", "Unknown"))

            with st.expander("🔍 View Raw LLM Output"):
                st.markdown(selected_data["Raw Output"])
    else:
        st.info("ℹ️ No resume match data found yet. Please run a match.")
else:
    st.info("ℹ️ Upload resumes and JDs, then click 'Match Resumes' to begin.")
