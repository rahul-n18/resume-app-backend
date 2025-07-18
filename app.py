from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK data is downloaded
for resource in ['stopwords', 'punkt']:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)

from nltk.corpus import stopwords

# 🛠 Setup FastAPI
app = FastAPI()

# Define your route handlers FIRST
class JDResume(BaseModel):
    resume: str
    jd: str

def calculate_match_percentage(text1, text2):
    vect = CountVectorizer().fit_transform([text1, text2]).toarray()
    return cosine_similarity(vect)[0,1] * 100

def extract_key_terms(text):
    stop = set(stopwords.words('english'))
    return set(w for w in text.lower().split() if w.isalpha() and w not in stop)

@app.post("/match_resume")
async def match_resume(payload: JDResume):
    res, jd = payload.resume or "", payload.jd or ""
    if not res or not jd:
        return {"error": "Need both resume and job description"}
    pct = calculate_match_percentage(res, jd)
    missing = list(extract_key_terms(jd) - extract_key_terms(res))
    if pct >= 70:
        suggestion = "✅ Good chances of being shortlisted."
    elif pct >= 40:
        suggestion = "⚠️ Good match, consider improvements." + (f" Missing: {', '.join(missing)}" if missing else "")
    else:
        suggestion = "❌ Poor match." + (f" Missing: {', '.join(missing)}" if missing else "")
    return {
        "match_percentage": pct,
        "missing_keywords": missing,
        "suggestion": suggestion
    }

# ✅ CORSMiddleware must be added **after** defining routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # allow all origins for dev/testing
    allow_credentials=False,     # cannot enable credentials with wildcard
    allow_methods=["*"],
    allow_headers=["*"],
)
