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

# ── ENDPOINT: GET LAST SAVED JD FOR USER ──────────────────────────────────────
@app.get("/last-jd/{user_id}")
async def get_last_jd(user_id: str):
    # find the most recent by timestamp
    cursor = jd_col.find({"user_id": user_id}).sort("timestamp", -1).limit(1)
    docs   = await cursor.to_list(length=1)
    if not docs:
        raise HTTPException(404, "No JD records found for this user")

    doc = docs[0]
    # convert and remove ObjectId
    doc_id = doc.pop("_id")
    doc["id"] = str(doc_id)
    return doc

# ── ENDPOINT: GET LAST SAVED CV FOR USER ─────────────────────────────────────
@app.get("/last-cv/{user_id}")
async def get_last_cv(user_id: str):
    cursor = cv_col.find({"user_id": user_id}).sort("timestamp", -1).limit(1)
    docs   = await cursor.to_list(length=1)
    if not docs:
        raise HTTPException(404, "No CV records found for this user")

    doc = docs[0]
    doc_id = doc.pop("_id")
    doc["id"] = str(doc_id)
    return doc

# ── RUN LOCAL ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
