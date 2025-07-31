# routers/match_skills.py

import os
import re
import json
import logging
from typing import List, Dict, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import motor.motor_asyncio
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("match_skills")

# ── DB SETUP ─────────────────────────────────────────────────────────────────
MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb+srv://yuvinsanketh10:EPveklyFAO7CeP3N"
    "@cluster0.5jwuszj.mongodb.net/"
    "job_analysis_db?retryWrites=true&w=majority&appName=Cluster0"
)
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
db = client["job_analysis_db"]
jd_col = db["jd_entities"]
cv_col = db["cv_skill_extraction"]

# ── OpenAI client setup ───────────────────────────────────────────────────────
# WARNING: fallback key is for local testing only. Move to env var in production.
OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    "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"
)
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required for question generation.")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ── EMBEDDING MODEL ──────────────────────────────────────────────────────────
model = SentenceTransformer("all-MiniLM-L6-v2")

# ── ROUTER ───────────────────────────────────────────────────────────────────
router = APIRouter(prefix="/match-skills", tags=["match-skills"])

# ── SCHEMAS ──────────────────────────────────────────────────────────────────
class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    answer: str  # correct answer

class MatchWithQuestions(BaseModel):
    technology: str
    score: float
    questions: List[QuizQuestion]
    error: Optional[str] = None

class MatchResponse(BaseModel):
    user_id: str
    matches: List[MatchWithQuestions]

class AnswerItem(BaseModel):
    technology: str
    question: str
    correct_answer: str
    user_answer: str

class EvaluateRequest(BaseModel):
    answers: List[AnswerItem]

# ── HELPERS ───────────────────────────────────────────────────────────────────
def parse_jd_entities(text: str) -> Dict[str, str]:
    pattern = r'^\s*(.+?)\s*->\s*(SKILL_[A-Z]+)\s*$'
    out: Dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(pattern, line)
        if m:
            skill, label = m.groups()
            out[skill.strip()] = label
    return out

def parse_resume_skills(skills: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for entry in skills:
        skill = entry.lstrip('-').strip()
        if skill and skill not in seen:
            seen.add(skill)
            out.append(skill)
    return out

def extract_json_array(raw: str) -> List[dict]:
    """
    Extract the first JSON array from model output and do minimal cleaning.
    """
    match = re.search(r"(\[.*\])", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON array found in model response.")
    json_str = match.group(1)

    def clean(s: str) -> str:
        if "'" in s and '"' not in s:
            s = s.replace("'", '"')
        s = re.sub(r",\s*([}\]])", r"\1", s)
        return s

    try:
        return json.loads(json_str)
    except Exception:
        cleaned = clean(json_str)
        return json.loads(cleaned)

def generate_questions_for_technology(technology: str, num_questions: int = 10) -> List[Dict]:
    """
    Use ChatGPT to generate MCQs (with correct answer) for a given technology.
    """
    system_prompt = "You are an expert software engineering quiz writer. Generate clear, concise multiple-choice questions."
    user_prompt = (
        f"Generate {num_questions} multiple-choice questions about \"{technology}\". "
        "Each question must have exactly 4 options. Include the correct answer in the field \"answer\". "
        "Respond with a single valid JSON array. Example element:\n"
        '{\n'
        '  "question": "What is JSX in React?",\n'
        '  "options": ["A syntax extension for JavaScript", "A database", "A CSS framework", "A build tool"],\n'
        '  "answer": "A syntax extension for JavaScript"\n'
        '}\n'
        "Do not include any explanation outside the array."
    )

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=1400,
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI request failed for {technology}: {e}")

    try:
        raw = resp.choices[0].message.content
    except Exception:
        raw = getattr(resp.choices[0].message, "content", "") if resp.choices else ""

    try:
        items = extract_json_array(raw)
    except Exception as e:
        raise RuntimeError(f"Failed to parse model output for {technology}: {e}; raw snippet: {raw[:1000]!r}")

    validated: List[Dict] = []
    for obj in items:
        q = obj.get("question")
        opts = obj.get("options")
        ans = obj.get("answer")
        if not (isinstance(q, str) and isinstance(opts, list) and len(opts) == 4 and isinstance(ans, str)):
            continue
        if ans not in opts:
            continue
        validated.append({
            "question": q.strip(),
            "options": [str(o).strip() for o in opts],
            "answer": ans.strip(),
        })
    return validated

async def fetch_latest_jd_and_cv(user_id: str):
    jd_docs = await jd_col.find({"user_id": user_id}).sort("timestamp", -1).limit(1).to_list(length=1)
    if not jd_docs:
        raise HTTPException(404, "No JD records found")
    cv_docs = await cv_col.find({"user_id": user_id}).sort("timestamp", -1).limit(1).to_list(length=1)
    if not cv_docs:
        raise HTTPException(404, "No CV records found")
    return jd_docs[0], cv_docs[0]

# ── ENDPOINT: MATCH SKILLS + QUESTION GENERATION ───────────────────────────────
@router.get("/{user_id}", response_model=MatchResponse)
async def match_skills_and_generate(
    user_id: str,
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Minimum similarity to include a match"),
    questions_per_skill: int = Query(10, ge=1, le=20, description="How many questions per matched technology")
):
    # 1. Fetch latest JD and CV
    jd_doc, cv_doc = await fetch_latest_jd_and_cv(user_id)

    # 2. Parse JD entities
    jd_list = [f"{e['text']} -> {e['label']}" for e in jd_doc.get("entities", [])]
    jd_map = parse_jd_entities("\n".join(jd_list))
    jd_skills = list(jd_map.keys())

    # 3. Parse CV skills
    raw_skills = cv_doc.get("skill_data", {}).get("all_skills", [])
    cv_skills = parse_resume_skills([f"- {s}" for s in raw_skills])

    if not jd_skills or not cv_skills:
        raise HTTPException(400, "Insufficient data to match")

    # 4. Semantic search
    jd_emb = model.encode(jd_skills, convert_to_tensor=True)
    cv_emb = model.encode(cv_skills, convert_to_tensor=True)
    hits = util.semantic_search(cv_emb, jd_emb, top_k=1)

    matches_out: List[MatchWithQuestions] = []

    for i, hit_list in enumerate(hits):
        if not hit_list:
            continue
        top = hit_list[0]
        score = float(top.get("score", 0.0))
        if score < threshold:
            continue
        technology = cv_skills[i]

        # Generate questions
        try:
            mcqs = generate_questions_for_technology(technology, num_questions=questions_per_skill)
            questions_schema = [
                QuizQuestion(question=q["question"], options=q["options"], answer=q["answer"])
                for q in mcqs
            ]
            matches_out.append(MatchWithQuestions(
                technology=technology,
                score=round(score, 3),
                questions=questions_schema
            ))
        except Exception as e:
            logger.exception("Failed generation for %s: %s", technology, e)
            matches_out.append(MatchWithQuestions(
                technology=technology,
                score=round(score, 3),
                questions=[],
                error=str(e)
            ))

    return MatchResponse(user_id=user_id, matches=matches_out)

# ── ENDPOINT: SUBMIT ANSWERS & SCORE ───────────────────────────────────────────
@router.post("/{user_id}/evaluate", response_model=Dict[str, Tuple[int, int]])
async def evaluate(user_id: str, req: EvaluateRequest):
    if not req.answers:
        raise HTTPException(400, "No answers provided.")
    tally: Dict[str, List[int]] = {}
    for ans in req.answers:
        tech = ans.technology.strip()
        correct, total = tally.get(tech, [0, 0])
        total += 1
        if ans.user_answer.strip().lower() == ans.correct_answer.strip().lower():
            correct += 1
        tally[tech] = [correct, total]
    return {tech: (counts[0], counts[1]) for tech, counts in tally.items()}
