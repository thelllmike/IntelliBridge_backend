# routers/quiz.py

import os
import re
import json
import math
from typing import List, Dict
from pathlib import Path
from collections import defaultdict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import motor.motor_asyncio
from sentence_transformers import SentenceTransformer, util
import joblib
import pandas as pd

# ── DATABASE SETUP ───────────────────────────────────────────────────────────
MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb+srv://yuvinsanketh10:EPveklyFAO7CeP3N"
    "@cluster0.5jwuszj.mongodb.net/job_analysis_db?"
    "retryWrites=true&w=majority&appName=Cluster0"
)
client   = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
db       = client["job_analysis_db"]
jd_col   = db["jd_entities"]
cv_col   = db["cv_skill_extraction"]

# ── LOAD MODELS & DATA ────────────────────────────────────────────────────────
rf         = joblib.load("rf_hire_model.joblib")
st_model   = SentenceTransformer("all-MiniLM-L6-v2")
MCQ_FILE   = Path(__file__).parent.parent / "mcq.json"
all_mcqs   = json.loads(MCQ_FILE.read_text(encoding="utf-8"))

# ── LABEL WEIGHTS ────────────────────────────────────────────────────────────
label_weights = {
    "SKILL_REQUIRED":   1.0,
    "SKILL_PREFERRED":  0.6,
    "SKILL_BONUS":      0.3,
}

# ── ROUTER DEFINITION ─────────────────────────────────────────────────────────
router = APIRouter(prefix="/quiz", tags=["quiz"])

# ── Pydantic SCHEMAS ─────────────────────────────────────────────────────────
class Question(BaseModel):
    id:       int
    skill:    str
    question: str
    options:  List[str]

class Answer(BaseModel):
    question_id: int
    answer:      str

class QuizResult(BaseModel):
    selected:        bool
    overall_percent: float           # average over participated skills
    breakdown:       Dict[str, float]  # only skills the user answered

# ── HELPERS ───────────────────────────────────────────────────────────────────
def parse_jd_entities(text: str) -> dict[str, str]:
    pat = r'^\s*(.+?)\s*->\s*(SKILL_[A-Z]+)\s*$'
    out = {}
    for ln in text.splitlines():
        m = re.match(pat, ln)
        if m:
            out[m.group(1).strip()] = m.group(2)
    return out

def parse_resume_skills(raw: List[str]) -> List[str]:
    seen = set()
    out  = []
    for e in raw:
        s = e.lstrip('-').strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out

async def fetch_latest_docs(user_id: str):
    jd_docs = await jd_col.find({"user_id": user_id})\
                         .sort("timestamp", -1).limit(1)\
                         .to_list(length=1)
    cv_docs = await cv_col.find({"user_id": user_id})\
                         .sort("timestamp", -1).limit(1)\
                         .to_list(length=1)
    if not jd_docs:
        raise HTTPException(404, "No JD found for this user")
    if not cv_docs:
        raise HTTPException(404, "No CV found for this user")
    return jd_docs[0], cv_docs[0]

# ── ENDPOINT: GET QUIZ QUESTIONS ──────────────────────────────────────────────
@router.get("/{user_id}", response_model=List[Question])
async def get_quiz(user_id: str):
    jd_doc, cv_doc = await fetch_latest_docs(user_id)

    # Build JD skill→label map
    jd_lines  = [f"{e['text']} -> {e['label']}" for e in jd_doc["entities"]]
    jd_map    = parse_jd_entities("\n".join(jd_lines))
    jd_skills = list(jd_map.keys())

    # Clean resume skills
    raw_skills  = cv_doc["skill_data"]["all_skills"]
    resume_list = parse_resume_skills([f"- {s}" for s in raw_skills])

    # Semantic matching
    jd_emb = st_model.encode(jd_skills, convert_to_tensor=True)
    cv_emb = st_model.encode(resume_list, convert_to_tensor=True)
    hits   = util.semantic_search(cv_emb, jd_emb, top_k=1)

    THRESH = 0.60
    matched = {
        resume_list[i]
        for i, h in enumerate(hits)
        if h and h[0]["score"] >= THRESH
    }

    # Build question list & sanitize options
    questions = []
    for idx, q in enumerate(all_mcqs):
        if q.get("skill") not in matched:
            continue
        clean_opts = []
        for opt in q.get("options", []):
            if isinstance(opt, str):
                clean_opts.append(opt)
            elif isinstance(opt, float) and math.isnan(opt):
                clean_opts.append("")
            else:
                clean_opts.append(str(opt))
        questions.append(
            Question(
                id=idx,
                skill=q.get("skill", ""),
                question=q.get("question", ""),
                options=clean_opts
            )
        )

    if not questions:
        raise HTTPException(404, "No MCQs found for matched skills")

    return questions

# ── ENDPOINT: SUBMIT QUIZ & PREDICT ───────────────────────────────────────────
@router.post("/{user_id}", response_model=QuizResult)
async def submit_quiz(user_id: str, answers: List[Answer]):
    jd_doc, cv_doc = await fetch_latest_docs(user_id)

    # Rebuild JD map
    jd_lines = [f"{e['text']} -> {e['label']}" for e in jd_doc["entities"]]
    jd_map   = parse_jd_entities("\n".join(jd_lines))

    # Recompute matched skills
    resume_raw  = cv_doc["skill_data"]["all_skills"]
    resume_list = parse_resume_skills([f"- {s}" for s in resume_raw])
    jd_emb   = st_model.encode(list(jd_map.keys()), convert_to_tensor=True)
    cv_emb   = st_model.encode(resume_list, convert_to_tensor=True)
    hits     = util.semantic_search(cv_emb, jd_emb, top_k=1)
    matched  = {
        resume_list[i]
        for i, h in enumerate(hits)
        if h and h[0]["score"] >= 0.50
    }

    # Map question IDs → MCQs
    quiz_map = {
        idx: q
        for idx, q in enumerate(all_mcqs)
        if q.get("skill") in matched
    }

    # Tally answers per skill
    tally = defaultdict(lambda: [0, 0])
    for ans in answers:
        q = quiz_map.get(ans.question_id)
        if not q:
            continue
        skill = q["skill"]
        tally[skill][1] += 1
        if ans.answer == q.get("correct_answer"):
            tally[skill][0] += 1

    # Compute breakdown and overall percentage over participated skills only
    breakdown = {}
    total_pct = 0
    count     = 0

    for skill, (correct, total_q) in tally.items():
        if total_q > 0:
            pct = (correct / total_q) * 100
            breakdown[skill] = round(pct, 1)
            total_pct += pct
            count += 1

    overall_pct = round(total_pct / count, 1) if count else 0.0
    selected    = overall_pct >= 30.0

    return QuizResult(
        selected=selected,
        overall_percent=overall_pct,
        breakdown=breakdown
    )
