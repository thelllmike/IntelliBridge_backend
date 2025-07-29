import os
from io import BytesIO
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import pdfplumber
import motor.motor_asyncio

from jd_analizer import analyze_text
from cv_analizer import extract_skills_from_file
from routers.user import router as user_router

# ── CONFIG ───────────────────────────────────────────────────────────────────
MONGODB_URI = (
    "mongodb+srv://yuvinsanketh10:EPveklyFAO7CeP3N"
    "@cluster0.5jwuszj.mongodb.net/"
    "job_analysis_db?retryWrites=true&w=majority&appName=Cluster0"
)
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
db     = client["job_analysis_db"]
jd_col = db["jd_entities"]
cv_col = db["cv_skill_extraction"]

# ── FASTAPI APP ──────────────────────────────────────────────────────────────
app = FastAPI()
app.include_router(user_router)

origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic MODELS ──────────────────────────────────────────────────────────
class JDRequest(BaseModel):
    user_id: str
    text:    str

# ── ENDPOINT: ANALYZE JD ─────────────────────────────────────────────────────
@app.post("/analyze-jd/")
async def analyze_jd(request: JDRequest):
    entities = analyze_text(request.text)
    now = datetime.utcnow()

    doc = {
        "user_id":    request.user_id,
        "input_text": request.text,
        "entities":   entities,
        "timestamp":  now,
    }
    result = await jd_col.insert_one(doc)
    return {"id": str(result.inserted_id), "entities": entities}

# ── ENDPOINT: UPLOAD & ANALYZE PDF ───────────────────────────────────────────
@app.post("/upload-pdf/")
async def upload_pdf(
    file:    UploadFile = File(...),
    user_id: str        = Form(...),
):
    if file.content_type != "application/pdf":
        raise HTTPException(400, "Only PDF files are supported.")

    content = await file.read()
    try:
        pdf = pdfplumber.open(BytesIO(content))
    except Exception as e:
        raise HTTPException(500, f"Failed to open PDF: {e}")

    base_name    = os.path.splitext(file.filename)[0]
    txt_filename = f"{base_name}.txt"
    txt_path     = Path(txt_filename)

    with txt_path.open("w", encoding="utf-8") as out_f:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            out_f.write(f"--- Page {i} ---\n{text}\n\n")
    pdf.close()

    skill_data = extract_skills_from_file(str(txt_path))
    now = datetime.utcnow()

    doc = {
        "user_id":    user_id,
        "filename":   file.filename,
        "txt_file":   txt_filename,
        "skill_data": skill_data,
        "timestamp":  now,
    }
    result = await cv_col.insert_one(doc)

    return JSONResponse({
        "id":               str(result.inserted_id),
        "message":          "Text extraction and skill analysis successful.",
        "txt_file":         txt_filename,
        "skill_extraction": skill_data,
    })


@app.get("/last-jd/{user_id}")
async def get_last_jd(user_id: str):
    # fetch the most recent JD doc
    cursor = jd_col.find({"user_id": user_id}).sort("timestamp", -1).limit(1)
    docs   = await cursor.to_list(length=1)
    if not docs:
        raise HTTPException(404, "No JD records found for this user")

    doc = docs[0]

    # 1) Pop off the ObjectId and convert to string
    _id = doc.pop("_id")
    doc["id"] = str(_id)

    # 2) Build your formatted lines
    #    e.g. ["Grid -> SKILL_PREFERRED", "Vue/Angular -> SKILL_PREFERRED", ...]
    formatted = [
        f"{ent['text']} -> {ent['label']}"
        for ent in doc.get("entities", [])
    ]

    # 3) Return both raw and formatted
    return {
        "id":                  doc["id"],
        "jd_skills":  formatted,
    
    }


@app.get("/last-cv/{user_id}")
async def get_last_cv(user_id: str):
    # Fetch the most recent CV record
    cursor = cv_col.find({"user_id": user_id}).sort("timestamp", -1).limit(1)
    docs   = await cursor.to_list(length=1)
    if not docs:
        raise HTTPException(404, "No CV records found for this user")

    doc = docs[0]
    # Remove the ObjectId and expose it as a string
    _id = doc.pop("_id")
    doc["id"] = str(_id)

    # Extract your list of skills
    all_skills = doc.get("skill_data", {}).get("all_skills", [])

    # Build the “- SkillName” list
    formatted_skills = [f"- {skill}" for skill in all_skills]

    return {
        "id":               doc["id"],
        "resume_skills": formatted_skills,
     
    }

@app.get("/last-all/{user_id}")
async def get_last_all(user_id: str):
    # 1) Load most recent JD
    jd_cursor = jd_col.find({"user_id": user_id}).sort("timestamp", -1).limit(1)
    jd_docs   = await jd_cursor.to_list(length=1)
    if not jd_docs:
        raise HTTPException(404, "No JD records found for this user")
    jd_doc = jd_docs[0]
    jd_id  = jd_doc.pop("_id")
    # format JD entities
    jd_formatted = [f"{ent['text']} -> {ent['label']}" for ent in jd_doc.get("entities", [])]

    # 2) Load most recent CV
    cv_cursor = cv_col.find({"user_id": user_id}).sort("timestamp", -1).limit(1)
    cv_docs   = await cv_cursor.to_list(length=1)
    if not cv_docs:
        raise HTTPException(404, "No CV records found for this user")
    cv_doc = cv_docs[0]
    cv_id  = cv_doc.pop("_id")
    # format CV skills
    cv_skills = cv_doc.get("skill_data", {}).get("all_skills", [])
    cv_formatted = [f"- {skill}" for skill in cv_skills]

    # 3) Return combined
    return {
        "jd": {
            
            "formatted":     jd_formatted,
            
        },
        "cv": {
            "formatted":     cv_formatted,
           
        }
    }
# ── RUN LOCAL ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
