from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from skills_extractor import SkillExtractor
import spacy

nlp = spacy.load('en_core_web_sm')
skill_extractor = SkillExtractor(nlp)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev, change to your domain for prod!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/extract_skills")
async def extract_skills(request: Request):
    data = await request.json()
    jd = data.get('jd', '')
    results = skill_extractor(jd)
    skills = results['results']['full_matches']  # list of skills found
    return { "skills": skills }
