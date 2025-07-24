# main.py
import os
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pdfplumber

app = FastAPI()

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    # 1) Validate file type
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    # 2) Read into memory
    content = await file.read()
    
    # 3) Open with pdfplumber
    try:
        pdf = pdfplumber.open(BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open PDF: {e}")
    
    # 4) Prepare output filename
    base_name = os.path.splitext(file.filename)[0]
    txt_filename = f"{base_name}.txt"
    
    # 5) Extract text page by page
    with open(txt_filename, "w", encoding="utf-8") as out_f:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            out_f.write(f"--- Page {i} ---\n")
            out_f.write(text + "\n\n")
    pdf.close()
    
    # 6) Return success
    return JSONResponse({
        "message": "Text extraction successful.",
        "txt_file": txt_filename
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
