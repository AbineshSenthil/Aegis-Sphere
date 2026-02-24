"""
Aegis-Sphere — Language Extraction (Phase 2)
SpaCy NER → Clinical Frame JSON (symptoms, meds, durations, conditions).
"""

import re
from typing import Optional


def extract_clinical_frame(transcript: Optional[str]) -> dict:
    """
    Extract structured clinical entities from a transcript using SpaCy + regex.

    Returns Clinical Frame JSON:
    {
        "symptoms": [...],
        "medications": [...],
        "durations": [...],
        "conditions": [...],
        "lab_values": [...],
        "vitals": [...],
        "demographics": {...}
    }
    """
    if not transcript:
        return _empty_frame()

    frame = {
        "symptoms": [],
        "medications": [],
        "durations": [],
        "conditions": [],
        "lab_values": [],
        "vitals": [],
        "demographics": {},
    }

    # ── Try SpaCy biomedical NER ──
    try:
        import spacy
        try:
            nlp = spacy.load("en_ner_bc5cdr_md")
        except OSError:
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                nlp = None

        if nlp:
            doc = nlp(transcript)
            for ent in doc.ents:
                label = ent.label_.upper()
                text = ent.text.strip()
                if label in ("DISEASE", "PROBLEM", "CONDITION"):
                    frame["conditions"].append(text)
                elif label in ("CHEMICAL", "DRUG", "MEDICATION"):
                    frame["medications"].append(text)
                elif label == "SYMPTOM":
                    frame["symptoms"].append(text)
    except ImportError:
        pass

    # ── Regex-based extraction (always runs as supplement) ──
    frame["symptoms"].extend(_extract_symptoms(transcript))
    frame["medications"].extend(_extract_medications(transcript))
    frame["durations"].extend(_extract_durations(transcript))
    frame["conditions"].extend(_extract_conditions(transcript))
    frame["lab_values"].extend(_extract_lab_values(transcript))
    frame["demographics"] = _extract_demographics(transcript)

    # Deduplicate
    for key in ["symptoms", "medications", "durations", "conditions", "lab_values"]:
        frame[key] = list(dict.fromkeys(frame[key]))  # preserves order

    return frame


def _empty_frame() -> dict:
    return {
        "symptoms": [],
        "medications": [],
        "durations": [],
        "conditions": [],
        "lab_values": [],
        "vitals": [],
        "demographics": {},
    }


def _extract_symptoms(text: str) -> list:
    symptom_patterns = [
        r"night sweats", r"weight loss", r"fever[s]?", r"cough(?:ing)?",
        r"lymphadenopathy", r"fatigue", r"dyspnea", r"shortness of breath",
        r"pain", r"nausea", r"vomiting", r"diarrhea", r"rash",
        r"bleeding", r"bruising", r"anorexia", r"malaise",
        r"swelling", r"headache", r"dizziness",
    ]
    found = []
    text_lower = text.lower()
    for pat in symptom_patterns:
        if re.search(pat, text_lower):
            match = re.search(pat, text_lower)
            found.append(match.group(0))
    return found


def _extract_medications(text: str) -> list:
    med_patterns = [
        r"tenofovir", r"lamivudine", r"dolutegravir", r"efavirenz",
        r"nevirapine", r"ritonavir", r"atazanavir", r"lopinavir",
        r"abacavir", r"zidovudine", r"emtricitabine",
        r"rifampicin", r"rifabutin", r"isoniazid", r"pyrazinamide",
        r"ethambutol", r"streptomycin",
        r"doxorubicin", r"cyclophosphamide", r"vincristine",
        r"prednisone", r"prednisolone", r"rituximab", r"methotrexate",
        r"cisplatin", r"carboplatin", r"paclitaxel", r"etoposide",
        r"cotrimoxazole", r"fluconazole", r"acyclovir",
        r"R-CHOP", r"CHOP", r"ART",
    ]
    found = []
    for pat in med_patterns:
        if re.search(pat, text, re.IGNORECASE):
            match = re.search(pat, text, re.IGNORECASE)
            found.append(match.group(0))
    return found


def _extract_durations(text: str) -> list:
    patterns = [
        r"\b\d+[\s-]*(week|month|day|year|hour)s?\b",
        r"\b(three|four|five|six|two|one)[\s-]*(week|month|day|year)s?\b",
    ]
    found = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            found.append(m.group(0).strip())
    return found


def _extract_conditions(text: str) -> list:
    cond_patterns = [
        r"HIV[- ]?positive", r"HIV\+?", r"lymphoma", r"Kaposi sarcoma",
        r"tuberculosis", r"\bTB\b", r"malaria", r"hepatitis",
        r"diabetes", r"hypertension", r"renal impairment",
        r"anemia", r"thrombocytopenia", r"neutropenia",
        r"pneumonia", r"meningitis", r"cancer",
        r"adenocarcinoma", r"cervical cancer",
    ]
    found = []
    for pat in cond_patterns:
        if re.search(pat, text, re.IGNORECASE):
            match = re.search(pat, text, re.IGNORECASE)
            found.append(match.group(0))
    return found


def _extract_lab_values(text: str) -> list:
    patterns = [
        r"CD4\s*(?:count\s*)?(?:of\s*|=\s*|was\s*)?\d+(?:\s*cells)?(?:\s*per\s*microliter)?",
        r"viral\s*load\s*(?:of\s*|=\s*)?\d+",
        r"hemoglobin\s*(?:of\s*|=\s*)?\d+\.?\d*",
        r"WBC\s*(?:of\s*|=\s*)?\d+\.?\d*",
        r"platelet[s]?\s*(?:of\s*|=\s*)?\d+",
        r"creatinine\s*(?:of\s*|=\s*)?\d+\.?\d*",
        r"eGFR\s*(?:<|>|=)?\s*\d+",
    ]
    found = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            found.append(m.group(0).strip())
    return found


def _extract_demographics(text: str) -> dict:
    demo = {}
    # Age
    age_match = re.search(r"(\d+)[\s-]*year[\s-]*old", text, re.IGNORECASE)
    if age_match:
        demo["age"] = int(age_match.group(1))
    # Sex
    if re.search(r"\b(male|man|gentleman)\b", text, re.IGNORECASE):
        demo["sex"] = "male"
    elif re.search(r"\b(female|woman|lady)\b", text, re.IGNORECASE):
        demo["sex"] = "female"
    return demo
