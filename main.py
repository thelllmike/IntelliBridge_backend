# main.py

import os
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import pdfplumber
from jd_analizer import analyze_text  # <— your new module
from cv_analizer import extract_skills_from_file

app = FastAPI()

# CORS setup (as before)
origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for JD analysis request
class JDRequest(BaseModel):
    text: str

@app.post("/analyze-jd/")
async def analyze_jd(request: JDRequest):
    """
    Analyze provided job description text and return detected entities.
    """
    entities = analyze_text(request.text)
    return {"entities": entities}
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    # Validate PDF
    if file.content_type != "application/pdf":
        raise HTTPException(400, "Only PDF files are supported.")

    # Read and extract text
    content = await file.read()
    try:
        pdf = pdfplumber.open(BytesIO(content))
    except Exception as e:
        raise HTTPException(500, f"Failed to open PDF: {e}")

    base_name   = os.path.splitext(file.filename)[0]
    txt_filename = f"{base_name}.txt"
    txt_path     = Path(txt_filename)

    with txt_path.open("w", encoding="utf-8") as out_f:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            out_f.write(f"--- Page {i} ---\n{text}\n\n")
    pdf.close()

    # Run the spaCy SKILL model on the extracted text
    skill_data = extract_skills_from_file(str(txt_path))

    # Return both the .txt filename and the skill results
    return JSONResponse({
        "message":          "Text extraction and skill analysis successful.",
        "txt_file":         txt_filename,
        "skill_extraction": skill_data,
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
