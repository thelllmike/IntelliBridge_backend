# routers/match_skills.py

import os
import re
from fastapi import APIRouter, HTTPException, Query
from sentence_transformers import SentenceTransformer, util
import motor.motor_asyncio

# ── DB SETUP ─────────────────────────────────────────────────────────────────
MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb+srv://yuvinsanketh10:EPveklyFAO7CeP3N"
    "@cluster0.5jwuszj.mongodb.net/"
    "job_analysis_db?retryWrites=true&w=majority&appName=Cluster0"
)
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
db     = client["job_analysis_db"]
jd_col = db["jd_entities"]
cv_col = db["cv_skill_extraction"]

# ── ROUTER & MODEL ───────────────────────────────────────────────────────────
router = APIRouter(prefix="/match-skills", tags=["match-skills"])
model  = SentenceTransformer('all-MiniLM-L6-v2')

# ── PARSING HELPERS ──────────────────────────────────────────────────────────
def parse_jd_entities(text: str) -> dict[str, str]:
    """Return dict {skill: label} from lines like 'X -> SKILL_Y'."""
    pattern = r'^\s*(.+?)\s*->\s*(SKILL_[A-Z]+)\s*$'
    out = {}
    for line in text.splitlines():
        m = re.match(pattern, line)
        if m:
            skill, label = m.groups()
            out[skill.strip()] = label
    return out

def parse_resume_skills(skills: list[str]) -> list[str]:
    """Dedupe and clean a list like ['- React', '- MySQL', …] → ['React','MySQL',…]."""
    seen = set()
    out  = []
    for entry in skills:
        skill = entry.lstrip('-').strip()
        if skill and skill not in seen:
            seen.add(skill)
            out.append(skill)
    return out

# ── ENDPOINT: MATCH SKILLS ───────────────────────────────────────────────────
@router.get("/{user_id}")
async def match_skills(
    user_id: str,
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Minimum similarity score to include a match")
):
    # 1) fetch latest JD
    jd_cursor = jd_col.find({"user_id": user_id}).sort("timestamp", -1).limit(1)
    jd_docs   = await jd_cursor.to_list(length=1)
    if not jd_docs:
        raise HTTPException(404, "No JD records found")
    jd_doc      = jd_docs[0]
    jd_list     = [f"{e['text']} -> {e['label']}" for e in jd_doc.get("entities", [])]
    jd_map      = parse_jd_entities("\n".join(jd_list))
    jd_skills   = list(jd_map.keys())

    # 2) fetch latest CV
    cv_cursor   = cv_col.find({"user_id": user_id}).sort("timestamp", -1).limit(1)
    cv_docs     = await cv_cursor.to_list(length=1)
    if not cv_docs:
        raise HTTPException(404, "No CV records found")
    cv_doc      = cv_docs[0]
    raw_skills  = cv_doc.get("skill_data", {}).get("all_skills", [])
    cv_skills   = parse_resume_skills([f"- {s}" for s in raw_skills])

    if not jd_skills or not cv_skills:
        raise HTTPException(400, "Insufficient data to match")

    # 3) encode & semantic search
    jd_emb   = model.encode(jd_skills, convert_to_tensor=True)
    cv_emb   = model.encode(cv_skills, convert_to_tensor=True)
    hits     = util.semantic_search(cv_emb, jd_emb, top_k=1)

    # 4) build results applying threshold
    matches = []
    for i, hit_list in enumerate(hits):
        if not hit_list:
            continue
        top = hit_list[0]
        score = float(top.get("score", 0.0))
        if score < threshold:
            continue  # skip below threshold
        jd_skill = jd_skills[top["corpus_id"]]
        matches.append({
            "technology": cv_skills[i],
        })

    return {"matches": matches}
