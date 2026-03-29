from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import json, os, re, uuid
from pathlib import Path
from typing import Optional
import numpy as np

# Optional imports
try:
    import pdfplumber
    PDF_SUPPORT = True
except:
    PDF_SUPPORT = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_SUPPORT = True
except:
    SKLEARN_SUPPORT = False

app = FastAPI(title="Resume Ranker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESUMES_FILE = DATA_DIR / "resumes.json"
DECISIONS_FILE = DATA_DIR / "decisions.json"

DATA_DIR.mkdir(exist_ok=True)

JOB_CATEGORIES = [
    "ACCOUNTANT", "ADVOCATE", "AGRICULTURE", "APPAREL", "ARTS",
    "AUTOMOBILE", "AVIATION", "BANKING", "BPO", "BUSINESS-DEVELOPMENT",
    "CHEF", "CONSTRUCTION", "CONSULTANT", "DESIGNER", "DIGITAL-MEDIA",
    "ENGINEERING", "FINANCE", "HEALTHCARE", "HR", "INFORMATION-TECHNOLOGY",
    "LAW", "MEDIA-ENTERTAINMENT", "SALES", "TEACHER"
]

def load_resumes():
    if RESUMES_FILE.exists():
        return json.loads(RESUMES_FILE.read_text())
    return []

def save_resumes(resumes):
    RESUMES_FILE.write_text(json.dumps(resumes, indent=2))

def load_decisions():
    if DECISIONS_FILE.exists():
        return json.loads(DECISIONS_FILE.read_text())
    return {}

def save_decisions(decisions):
    DECISIONS_FILE.write_text(json.dumps(decisions, indent=2))

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_from_pdf(content: bytes) -> str:
    if not PDF_SUPPORT:
        return ""
    import io
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            return " ".join(page.extract_text() or "" for page in pdf.pages)
    except:
        return ""

def calculate_match_score(job_desc: str, resume_text: str) -> float:
    job_lower = job_desc.lower()
    resume_lower = resume_text.lower()

    # Keyword match
    job_keywords = set(w for w in re.findall(r'\b\w{4,}\b', job_lower))
    resume_words = set(re.findall(r'\b\w{4,}\b', resume_lower))
    if not job_keywords:
        return 50.0
    keyword_score = len(job_keywords & resume_words) / len(job_keywords) * 100

    # TF-IDF cosine similarity
    tfidf_score = 0.0
    if SKLEARN_SUPPORT:
        try:
            vec = TfidfVectorizer(max_features=500, stop_words='english')
            matrix = vec.fit_transform([job_desc, resume_text])
            tfidf_score = cosine_similarity(matrix[0], matrix[1])[0][0] * 100
        except:
            pass

    # Weighted combination with a floor
    raw = (keyword_score * 0.6) + (tfidf_score * 0.4)
    final = raw * 0.65 + 35
    return round(min(100, max(0, final)), 1)

# ── Routes ──────────────────────────────────────────────

@app.get("/api/categories")
def get_categories():
    return {"categories": JOB_CATEGORIES}

@app.post("/api/upload")
async def upload_resume(
    file: UploadFile = File(...),
    category: str = Form(...),
    candidate_name: Optional[str] = Form(None)
):
    if category not in JOB_CATEGORIES:
        raise HTTPException(400, f"Invalid category: {category}")

    content = await file.read()
    filename = file.filename or "unknown"

    if filename.lower().endswith(".pdf"):
        text = extract_text_from_pdf(content)
        if not text.strip():
            raise HTTPException(400, "Could not extract text from PDF. Try a text-based PDF.")
    elif filename.lower().endswith(".txt"):
        text = content.decode("utf-8", errors="ignore")
    else:
        raise HTTPException(400, "Only PDF and TXT files are supported.")

    text = clean_text(text)
    if len(text) < 50:
        raise HTTPException(400, "Resume text too short (< 50 characters).")

    resume_id = str(uuid.uuid4())[:8]
    name = candidate_name.strip() if candidate_name and candidate_name.strip() else filename.rsplit(".", 1)[0]

    resumes = load_resumes()
    resumes.append({
        "id": resume_id,
        "name": name,
        "filename": filename,
        "category": category,
        "text": text[:6000],
        "length": len(text)
    })
    save_resumes(resumes)

    return {"success": True, "resume_id": resume_id, "name": name, "category": category}

@app.get("/api/search")
def search_resumes(category: str, job_description: str = ""):
    resumes = load_resumes()
    decisions = load_decisions()

    category_resumes = [r for r in resumes if r["category"] == category]
    if not category_resumes:
        return {"results": [], "total": 0}

    jd = job_description.strip() or f"Looking for a skilled {category.lower()} professional with relevant experience."

    results = []
    for r in category_resumes:
        score = calculate_match_score(jd, r["text"])
        results.append({
            "id": r["id"],
            "name": r["name"],
            "filename": r["filename"],
            "category": r["category"],
            "score": score,
            "preview": r["text"][:400] + "..." if len(r["text"]) > 400 else r["text"],
            "decision": decisions.get(r["id"], "pending"),
            "length": r["length"]
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return {"results": results, "total": len(results)}

@app.post("/api/decision")
def set_decision(payload: dict):
    resume_id = payload.get("resume_id")
    decision = payload.get("decision")  # shortlist | waitlist | reject | pending

    if not resume_id or decision not in ("shortlist", "waitlist", "reject", "pending"):
        raise HTTPException(400, "Invalid payload")

    decisions = load_decisions()
    decisions[resume_id] = decision
    save_decisions(decisions)
    return {"success": True, "resume_id": resume_id, "decision": decision}

@app.get("/api/decisions")
def get_all_decisions():
    return load_decisions()

@app.delete("/api/resume/{resume_id}")
def delete_resume(resume_id: str):
    resumes = load_resumes()
    resumes = [r for r in resumes if r["id"] != resume_id]
    save_resumes(resumes)
    decisions = load_decisions()
    decisions.pop(resume_id, None)
    save_decisions(decisions)
    return {"success": True}

@app.get("/api/stats")
def get_stats():
    resumes = load_resumes()
    decisions = load_decisions()
    by_category = {}
    for r in resumes:
        by_category[r["category"]] = by_category.get(r["category"], 0) + 1
    return {
        "total_resumes": len(resumes),
        "by_category": by_category,
        "shortlisted": sum(1 for d in decisions.values() if d == "shortlist"),
        "rejected": sum(1 for d in decisions.values() if d == "reject"),
        "waitlisted": sum(1 for d in decisions.values() if d == "waitlist"),
    }

# Serve frontend
FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
