# routers/match_skills.py

import os
import re
import json
import logging
from typing import List, Dict, Optional, Tuple, Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse
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

# ── OpenRouter / OpenAI client setup ──────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY",
    "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"
)
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY environment variable is required.")

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324:free")

openai_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

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
        return json.loads(clean(json_str))

def generate_questions_for_technology(technology: str, num_questions: int = 10) -> List[Dict]:
    system_prompt = "You are an expert software engineering quiz writer. Generate clear, concise multiple-choice questions."
    user_prompt = (
        f"Generate {num_questions} multiple-choice questions about \"{technology}\". "
        "Each question must have exactly 4 options. Include the correct answer in the field \"answer\". "
        "Respond with a single valid JSON array of objects with keys: question, options, answer. "
        "Example:\n"
        '{\n'
        '  "question": "What is JSX in React?",\n'
        '  "options": ["A syntax extension for JavaScript", "A database", "A CSS framework", "A build tool"],\n'
        '  "answer": "A syntax extension for JavaScript"\n'
        '}\n'
        "Do not include any explanation outside the JSON array."
    )

    try:
        resp = openai_client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=1400,
            extra_headers={
                "HTTP-Referer": "https://yourapp.example",
                "X-Title": "MatchSkillsQuiz",
            },
        )
    except Exception as e:
        raise RuntimeError(f"OpenRouter request failed for {technology}: {e}")

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

def normalize_answer(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def fallback_grade(answers: List[AnswerItem]) -> Tuple[Dict[str, Tuple[int, int]], List[dict]]:
    per_skill: Dict[str, Tuple[int, int]] = {}
    details = []
    for ans in answers:
        tech = ans.technology.strip()
        corr, tot = per_skill.get(tech, (0, 0))
        tot += 1
        is_correct = normalize_answer(ans.user_answer) == normalize_answer(ans.correct_answer)
        if is_correct:
            corr += 1
        per_skill[tech] = (corr, tot)
        details.append({
            "technology": tech,
            "question": ans.question,
            "correct_answer": ans.correct_answer,
            "user_answer": ans.user_answer,
            "is_correct": is_correct,
            "reason": "exact match" if is_correct else "mismatch (fallback)"
        })
    return per_skill, details

def grade_with_llm(answers: List[AnswerItem]) -> Tuple[Dict[str, Tuple[int, int]], List[dict]]:
    # Build a compact representation for the model
    payload = []
    for a in answers:
        payload.append({
            "technology": a.technology,
            "question": a.question,
            "correct_answer": a.correct_answer,
            "user_answer": a.user_answer,
        })
    system_prompt = (
        "You are a precise quiz grader. For each provided item, compare the user's answer to the correct answer. "
        "Determine whether the user's answer is correct. Accept synonyms and minor variations if meaning is preserved. "
        "Respond with a single JSON object with two keys: "
        "'summary' mapping technology to {'correct': int, 'total': int, 'percentage': float}, "
        "and 'details' which is an array of objects with fields: technology, question, correct_answer, user_answer, is_correct (true/false), and a brief reason. "
        "Do not include any extraneous text outside the JSON."
    )
    user_prompt = f"Answers:\n{json.dumps(payload, ensure_ascii=False)}"

    try:
        resp = openai_client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,  # deterministic grading
            max_tokens=1200,
            extra_headers={
                "HTTP-Referer": "https://yourapp.example",
                "X-Title": "MatchSkillsGrader",
            },
        )
        raw = resp.choices[0].message.content
        parsed = json.loads(raw)
        summary = parsed.get("summary", {})
        details = parsed.get("details", [])
        per_skill: Dict[str, Tuple[int, int]] = {}
        # Normalize summary into tuples
        for tech, v in summary.items():
            correct = int(v.get("correct", 0))
            total = int(v.get("total", 0))
            per_skill[tech] = (correct, total)
        # If LLM didn't supply expected summary, fallback to computing from details
        if not per_skill and isinstance(details, list):
            for d in details:
                tech = d.get("technology", "")
                is_correct = bool(d.get("is_correct", False))
                corr, tot = per_skill.get(tech, (0, 0))
                tot += 1
                if is_correct:
                    corr += 1
                per_skill[tech] = (corr, tot)
        return per_skill, details
    except Exception as e:
        logger.warning("LLM grading failed, falling back: %s", e)
        return fallback_grade(answers)

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
    jd_doc, cv_doc = await fetch_latest_jd_and_cv(user_id)

    jd_list = [f"{e['text']} -> {e['label']}" for e in jd_doc.get("entities", [])]
    jd_map = parse_jd_entities("\n".join(jd_list))
    jd_skills = list(jd_map.keys())

    raw_skills = cv_doc.get("skill_data", {}).get("all_skills", [])
    cv_skills = parse_resume_skills([f"- {s}" for s in raw_skills])

    if not jd_skills or not cv_skills:
        raise HTTPException(400, "Insufficient data to match")

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
        try:
            mcqs = generate_questions_for_technology(technology, num_questions=questions_per_skill)
            questions_schema = [QuizQuestion(**q) for q in mcqs]
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

# ── ENDPOINT: SUBMIT ANSWERS & GRADE VIA LLM ───────────────────────────────────
@router.post("/{user_id}/evaluate")
async def evaluate(
    user_id: str,
    req: EvaluateRequest,
    compact: bool = Query(False, description="If true, return only training-skill style tuples in plain text")
):
    if not req.answers:
        raise HTTPException(400, "No answers provided.")

    # Grade answers via Deepseek LLM (with fallback)
    per_skill, details = grade_with_llm(req.answers)

    # Build top_skills sorted descending by percentage
    top_skills = []
    total_percent = 0.0
    count = 0
    for tech, (correct, total) in per_skill.items():
        pct = (correct / total * 100) if total > 0 else 0.0
        top_skills.append({"technology": tech, "percentage": round(pct, 1)})
        total_percent += pct
        count += 1
    top_skills.sort(key=lambda x: -x["percentage"])

    overall_avg = round(total_percent / count, 1) if count else 0.0
    selected = overall_avg >= 50.0  # simple threshold

    if compact:
        # produce lines like: "React": (5, 10),   # 50%
        lines = []
        max_skill_len = max((len(s) for s in per_skill.keys()), default=0)
        for skill, (correct, total) in per_skill.items():
            pct = (correct / total * 100) if total > 0 else 0.0
            skill_field = f'"{skill}":'
            tuple_field = f"({correct}, {total})"
            padded_skill = skill_field.ljust(max_skill_len + 3)
            line = f"{padded_skill} {tuple_field},   # {round(pct)}%"
            lines.append(line)
        body = "\n".join(lines)
        return PlainTextResponse(content=body, media_type="text/plain")

    response = {
        "per_skill": {tech: (correct, total) for tech, (correct, total) in per_skill.items()},
        "selected": selected,
        "average_percentage": overall_avg,
        "top_skills": top_skills,
        "details": details,  # each question-level evaluation
    }
    return response
