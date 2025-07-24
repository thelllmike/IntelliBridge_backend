# cv_analizer.py

import spacy
from pathlib import Path

# 1) Point to where your model lives
MODEL_DIR = Path(__file__).parent / "models" / "cvoutput" / "model-best"

# 2) Load once
cv_nlp = spacy.load(str(MODEL_DIR))

def extract_skills_from_file(txt_path: str) -> dict:
    """
    Read the given .txt file, run the spaCy SKILL model,
    and return both the full list of detected skills and the unique set.
    """
    text = Path(txt_path).read_text(encoding="utf-8")
    doc = cv_nlp(text)

    skills = [ent.text for ent in doc.ents if ent.label_ == "SKILL"]
    unique_skills = list(set(skills))

    return {
        "all_skills": skills,
        "unique_skills": unique_skills,
        "count": len(unique_skills),
    }
