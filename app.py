from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import spacy
from spacy.matcher import PhraseMatcher

# Download spaCy model at runtime if not present
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Replace with a bigger/better list as you like
skills_list = [
    "python", "sql", "aws", "docker", "project management", "linux", "javascript",
    "machine learning", "excel", "git", "communication", "pandas", "tensorflow"
]
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp.make_doc(skill) for skill in skills_list]
matcher.add("SKILLS", patterns)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/extract_skills")
async def extract_skills(request: Request):
    data = await request.json()
    jd = data.get('jd', '')
    doc = nlp(jd)
    matches = matcher(doc)
    found_skills = sorted(set([doc[start:end].text for match_id, start, end in matches]))
    return {"skills": found_skills}
