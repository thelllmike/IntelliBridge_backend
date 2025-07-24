# jd_analizer.py

import spacy
from pathlib import Path

# point to your spaCy export under models/jdoutput/model-best
MODEL_DIR = Path(__file__).parent / "models" / "jdoutput" / "model-best"

# load once at import time
nlp = spacy.load(str(MODEL_DIR))

def analyze_text(text: str):
    """
    Run the spaCy NER on the given text and return a list of entity dicts.
    """
    doc = nlp(text)
    return [
        {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        }
        for ent in doc.ents
    ]
