import os
import re
import requests
from docx import Document
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()  # Load .env
# Load text from job description/resume
def load_text_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# llm_util.py

def build_prompt(jd_text, resume_text):
    return f"""
You are a senior recruiter evaluating resumes.

Evaluate the resume below against the job description and return a structured analysis with these sections:

- 🎯 Match Score (out of 100)
- 🎓 Resume Score (out of 10)
- 📝 3-line Summary
- ✅ Strengths (skills, tools, experience)
- 🚨 Weaknesses (gaps, missing experience or skills)
- 🛠 Suggestions to improve
- 🏅 Badges (✅ or ❌ style for skills)
- 📋 Summary Table of Scores (markdown format with categories and scores)
- 🧠 Chain-of-thought reflection
- ✨ Unique Differentiators
- 📢 Final Verdict (Should they be interviewed?)

--- JOB DESCRIPTION ---
{jd_text}

--- RESUME ---
{resume_text}
"""


# Call Together API
def analyze_with_together(prompt):
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")  # Make sure this is set in your environment
    url = "https://api.together.xyz/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "meta-llama/Llama-3-70b-chat-hf",
        "messages": [
            {"role": "system", "content": "You are a helpful and honest assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1500
    }

    response = requests.post(url, headers=headers, json=payload)
    result = response.json()
    return result['choices'][0]['message']['content']


import re

def extract_llm_output(raw_output, resume_name, jd_name):
    def extract_section(regex, default=None):
        match = re.search(regex, raw_output, re.DOTALL)
        return match.group(1).strip() if match else default

    parsed_data = {
        "Resume": resume_name,
        "Job Description": jd_name,
        "Match Score": int(extract_section(r"🎯 Match Score: (\d+)", 0)),
        "Resume Score": int(extract_section(r"🎓 Resume Score: (\d+)", 0)),
        "Summary": extract_section(r"📝 3-line Summary\s*:?(.+?)(?=\n[A-Z💡🔎🎯✅🚨🛠🏅📋🧠✨📢]|$)", ""),
        "Strengths": extract_section(r"✅ Strengths\s*:?(.+?)(?=\n[A-Z💡🔎🎯🚨🛠🏅📋🧠✨📢]|$)", ""),
        "Weaknesses": extract_section(r"🚨 Weaknesses\s*:?(.+?)(?=\n[A-Z💡🔎🎯✅🛠🏅📋🧠✨📢]|$)", ""),
        "Verdict": extract_section(r"📢 (?:Final )?Verdict\s*:?(.+)", "Unknown"),
        "Raw Output": raw_output
    }

    # Optional: Extract more (badges, summary table, etc.)

    return parsed_data

