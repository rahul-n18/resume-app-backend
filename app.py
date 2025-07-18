from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data at startup if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.corpus import stopwords

# --- FastAPI setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # Allow all origins
    allow_credentials=False, # Credentials must be off for wildcard origin
    allow_methods=["*"],
    allow_headers=["*"],
)


class JDResume(BaseModel):
    resume: str
    jd: str

def calculate_match_percentage(text1, text2):
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    match_percentage = cosine_sim[0, 1] * 100
    return match_percentage

def extract_key_terms(text):
    stop_words = set(stopwords.words('english'))
    tokens = text.lower().split()
    filtered_words = [w for w in tokens if w not in stop_words and w.isalpha()]
    return set(filtered_words)

@app.post("/match_resume")
async def match_resume(payload: JDResume):
    resume_text = payload.resume or ""
    jd_text = payload.jd or ""
    if not resume_text or not jd_text:
        return {
            "error": "Please enter both Resume and Job Description."
        }

    match_percentage = calculate_match_percentage(resume_text, jd_text)
    jd_terms = extract_key_terms(jd_text)
    resume_terms = extract_key_terms(resume_text)
    missing_terms = list(jd_terms - resume_terms)

    if match_percentage >= 70:
        suggestion = "✅ Good Chances of getting your Resume Shortlisted."
    elif 40 <= match_percentage < 70:
        suggestion = "⚠️ Good match but can be improved."
        if missing_terms:
            suggestion += f"\nConsider adding these key terms from the job description to your resume: {', '.join(missing_terms)}"
    else:
        suggestion = "❌ Poor match."
        if missing_terms:
            suggestion += f"\nYour resume is missing these key terms from the job description: {', '.join(missing_terms)}"

    return {
        "match_percentage": match_percentage,
        "missing_keywords": missing_terms,
        "suggestion": suggestion
    }
