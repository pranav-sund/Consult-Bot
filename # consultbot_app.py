# consultbot_app.py
import os
import re
import json
import sqlite3
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Request, Body, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import pandas as pd
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from pathlib import Path
import uuid
import shutil

# Optional/conditional imports (embeddings)
USE_LANGCHAIN = True
try:
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.text_splitter import CharacterTextSplitter
    import faiss
except Exception:
    # If not installed, embedding mode will be disabled at runtime
    USE_LANGCHAIN = False

# -------- CONFIG: role parameters from your mini-project --------
SALESPERSON_NAME = os.getenv("SALESPERSON_NAME", "ConsultBot")
SALESPERSON_ROLE = os.getenv("SALESPERSON_ROLE", "Client Support Assistant")
COMPANY_NAME = os.getenv("COMPANY_NAME", "Consultancy & Advisory")
COMPANY_BUSINESS = os.getenv(
    "COMPANY_BUSINESS",
    "Consultancy & Advisory: Business strategy, brand development, digital marketing and operational optimization consulting."
)
COMPANY_VALUES = os.getenv(
    "COMPANY_VALUES",
    "Deliver accurate, timely, and professional advisory support backed by data and domain experience."
)

CSV_PATH = "/mnt/data/Chatbot_System_Merged_30Rows.csv"  # provided by user
DATA_DIR = Path("./consultbot_data")
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "consultbot.db"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index"  # if embeddings used, saved here
FAISS_INDEX_PATH.mkdir(exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -------- FastAPI app --------
app = FastAPI(title="ConsultBot - Consultancy & Advisory Chatbot")

# -------- Utilities: load CSV and prepare knowledge base --------
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at {path}")
    df = pd.read_csv(path)
    # Ensure 'Question' and 'Answer' columns exist (case-insensitive)
    cols = {c.lower(): c for c in df.columns}
    if 'question' not in cols or 'answer' not in cols:
        raise ValueError("CSV must contain 'Question' and 'Answer' columns.")
    df = df.rename(columns={cols['question']: 'Question', cols['answer']: 'Answer'})
    df = df[['Question', 'Answer']].dropna().drop_duplicates().reset_index(drop=True)
    return df

KB_DF = load_csv(CSV_PATH)

# Build simple in-memory FAQ list for rule-based mode
FAQ_LIST = [{"id": idx, "question": str(row.Question).strip(), "answer": str(row.Answer).strip()} 
            for idx, row in KB_DF.iterrows()]

# -------- Simple fuzzy match for rule-based mode --------
import difflib
def rule_based_answer(query: str, top_n: int = 3) -> Dict[str, Any]:
    q = query.lower().strip()
    # exact match first
    for faq in FAQ_LIST:
        if faq["question"].lower().strip() == q:
            return {"mode": "rule", "match_type": "exact", "answer": faq["answer"], "faq_id": faq["id"]}
    # use difflib for similarity
    questions = [f["question"] for f in FAQ_LIST]
    matches = difflib.get_close_matches(query, questions, n=top_n, cutoff=0.5)
    if matches:
        matched = matches[0]
        faq = next(f for f in FAQ_LIST if f["question"] == matched)
        return {"mode": "rule", "match_type": "fuzzy", "score_example": None, "answer": faq["answer"], "faq_id": faq["id"], "matched_question": matched}
    # fallback: return top-k possible questions to clarify
    suggestions = questions[:min(5, len(questions))]
    return {"mode": "rule", "match_type": "none", "answer": "I couldn't find an exact answer. Here are related questions you can try: " + " | ".join(suggestions)}

# -------- Embeddings support (Mode B & optionally C) --------
EMBEDDINGS_AVAILABLE = False
EMBEDDING_STORE = None
EMBEDDING_TEXTS = None

def get_embedder():
    global EMBEDDINGS_AVAILABLE, EMBEDDING_STORE, EMBEDDING_TEXTS
    if not USE_LANGCHAIN or not OPENAI_API_KEY:
        EMBEDDINGS_AVAILABLE = False
        return None
    try:
        embedder = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        # Build or load FAISS index
        # If already built, load it
        index_file = FAISS_INDEX_PATH / "faiss_store.pkl"
        if index_file.exists():
            EMBEDDING_STORE = FAISS.load_local(str(FAISS_INDEX_PATH), embedder)
            EMBEDDINGS_AVAILABLE = True
            return embedder
        # Build store
        EMBEDDING_TEXTS = [f"{faq['question']}\n\n{faq['answer']}" for faq in FAQ_LIST]
        # Splitting not strictly necessary for short Q/A, but keep it simple
        EMBEDDING_STORE = FAISS.from_texts(EMBEDDING_TEXTS, embedder)
        # persist to disk
        EMBEDDING_STORE.save_local(str(FAISS_INDEX_PATH))
        EMBEDDINGS_AVAILABLE = True
        return embedder
    except Exception as e:
        print("Embeddings not available:", e)
        EMBEDDINGS_AVAILABLE = False
        return None

# Try to initialize embeddings at startup (optional)
_embedder = get_embedder()

def embedding_search_answer(query: str, k: int = 3) -> Dict[str, Any]:
    if not EMBEDDINGS_AVAILABLE or EMBEDDING_STORE is None:
        return {"mode": "embed", "error": "Embeddings not available. Set OPENAI_API_KEY and install dependencies."}
    # retrieve
    docs_and_scores = EMBEDDING_STORE.similarity_search_with_score(query, k=k)
    # docs_and_scores => list of (doc, score)
    best_docs = []
    for doc, score in docs_and_scores:
        # doc.page_content corresponds to the text we stored: question \n\n answer
        content = doc.page_content
        if "\n\n" in content:
            q,a = content.split("\n\n",1)
        else:
            q,a = content, ""
        best_docs.append({"question": q.strip(), "answer": a.strip(), "score": float(score)})
    return {"mode": "embed", "results": best_docs}

# -------- SQLite chat logs & appointment DB (Mode C) --------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS chat_logs (
        id TEXT PRIMARY KEY,
        user_name TEXT,
        user_email TEXT,
        timestamp TEXT,
        query TEXT,
        response TEXT,
        mode TEXT
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS appointments (
        id TEXT PRIMARY KEY,
        client_name TEXT,
        client_email TEXT,
        date TEXT,
        time TEXT,
        status TEXT,
        created_at TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()

def log_chat(user_name: str, user_email: str, query: str, response: str, mode: str = "rule"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO chat_logs (id, user_name, user_email, timestamp, query, response, mode) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (str(uuid.uuid4()), user_name, user_email, datetime.utcnow().isoformat(), query, response, mode))
    conn.commit()
    conn.close()

def create_appointment(client_name: str, client_email: str, date: str, time_str: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    app_id = str(uuid.uuid4())
    c.execute("INSERT INTO appointments (id, client_name, client_email, date, time, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (app_id, client_name, client_email, date, time_str, "confirmed", datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()
    return app_id

def list_appointments():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, client_name, client_email, date, time, status, created_at FROM appointments ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    keys = ["id", "client_name", "client_email", "date", "time", "status", "created_at"]
    return [dict(zip(keys, r)) for r in rows]

# -------- PDF generation for chat summary --------
def generate_chat_pdf(user_name: str, user_email: str, chat_entries: List[Dict[str, str]], dest_path: Path):
    c = canvas.Canvas(str(dest_path), pagesize=letter)
    width, height = letter
    y = height - 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, f"Chat Summary - {COMPANY_NAME}")
    y -= 30
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Client: {user_name} <{user_email}>")
    y -= 20
    c.drawString(40, y, f"Generated: {datetime.utcnow().isoformat()} UTC")
    y -= 30
    for entry in chat_entries:
        if y < 80:
            c.showPage()
            y = height - 40
        ts = entry.get("timestamp", "")
        q = entry.get("query", "")
        r = entry.get("response", "")
        c.setFont("Helvetica-Bold", 10)
        c.drawString(40, y, f"[{ts}] User: {q}")
        y -= 14
        c.setFont("Helvetica", 10)
        c.drawString(60, y, f"Bot: {r}")
        y -= 18
    c.showPage()
    c.save()

# -------- Pydantic models for requests --------
class ChatRequest(BaseModel):
    user_name: Optional[str] = "Guest"
    user_email: Optional[str] = "guest@example.com"
    query: str
    mode: Optional[str] = "rule"  # rule | embed | full
    use_embeddings: Optional[bool] = True  # only meaningful for embed/full
    # allow extra parameters for SalesGPT-like behaviour
    salesperson_name: Optional[str] = SALESPERSON_NAME
    company_name: Optional[str] = COMPANY_NAME

class AppointmentRequest(BaseModel):
    client_name: str
    client_email: str
    date: str  # YYYY-MM-DD
    time: str  # HH:MM

# -------- Main endpoint: /consultbot --------
@app.post("/consultbot")
async def consultbot_chat(req: ChatRequest):
    """
    Main chat endpoint.
    mode: 'rule' -> exact/fuzzy rule-based over CSV
          'embed' -> embeddings semantic search (requires OPENAI_API_KEY and dependencies)
          'full'  -> full system behaviors: embedding (optional), appointment suggestions, PDF etc.
    """
    query = req.query.strip()
    mode = (req.mode or "rule").lower()
    use_embeddings = bool(req.use_embeddings)

    # Basic instruction: if the user asks to book an appointment, route to appointment creation / suggestion (simple parsing)
    booking_intent = False
    if re.search(r"\b(book|appointment|meet|schedule|slot)\b", query, re.I):
        booking_intent = True

    # Mode: rule-based
    if mode == "rule":
        result = rule_based_answer(query)
        # log chat
        log_chat(req.user_name, req.user_email, query, result.get("answer", ""), mode="rule")
        return JSONResponse({"status": "ok", "mode": "rule", "result": result})

    # Mode: embeddings
    elif mode == "embed":
        if not USE_LANGCHAIN or not OPENAI_API_KEY or not EMBEDDINGS_AVAILABLE:
            return JSONResponse({"status": "error", "message": "Embeddings not available. Set OPENAI_API_KEY and install requirements (langchain, faiss).", "available": EMBEDDINGS_AVAILABLE}, status_code=400)
        emb_res = embedding_search_answer(query, k=3)
        # Compose LLM-style response optionally using the top answer(s)
        # For safety, return the top matched answer and source questions
        top = emb_res.get("results", [])[0] if emb_res.get("results") else None
        answer_text = top["answer"] if top else "I could not find a good match."
        log_chat(req.user_name, req.user_email, query, answer_text, mode="embed")
        return JSONResponse({"status": "ok", "mode": "embed", "results": emb_res})

    # Mode: full (FAQ + appointment + admin features)
    elif mode == "full":
        # If embeddings requested and available, use embeddings for higher-quality semantic match, else fallback to rule
        answer = None
        used = None
        if use_embeddings and USE_LANGCHAIN and OPENAI_API_KEY and EMBEDDINGS_AVAILABLE:
            emb_res = embedding_search_answer(query, k=2)
            results = emb_res.get("results", [])
            if results:
                # we can prefer highest score (closest)
                answer = results[0]["answer"]
                used = "embed"
        if not answer:
            # fallback to rule based
            rb = rule_based_answer(query)
            answer = rb.get("answer")
            used = "rule"

        # If booking intent, create a tentative appointment (simple auto-suggest); in production you'd confirm date/time with user
        appointment_id = None
        if booking_intent:
            # try to extract a date in iso format; naive extraction:
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", query)
            time_match = re.search(r"(\d{1,2}:\d{2})", query)
            if date_match and time_match:
                try:
                    app_id = create_appointment(req.user_name, req.user_email, date_match.group(1), time_match.group(1))
                    appointment_id = app_id
                    answer += f"\n\nI've scheduled an appointment for you on {date_match.group(1)} at {time_match.group(1)} (ID: {app_id}). A confirmation has been recorded."
                except Exception as e:
                    answer += f"\n\nI attempted to create an appointment but encountered an error: {e}"
            else:
                # prompt for date/time
                answer += "\n\nI can help book an appointment. Please tell me preferred date (YYYY-MM-DD) and time (HH:MM)."

        # log chat
        log_chat(req.user_name, req.user_email, query, answer, mode="full")
        return JSONResponse({"status": "ok", "mode": "full", "used": used, "answer": answer, "appointment_id": appointment_id})
    else:
        raise HTTPException(status_code=400, detail="Invalid mode. Use 'rule', 'embed', or 'full'.")

# -------- Endpoint: create appointment directly (Mode C) --------
@app.post("/appointments/create")
async def create_appointment_endpoint(payload: AppointmentRequest):
    try:
        app_id = create_appointment(payload.client_name, payload.client_email, payload.date, payload.time)
        return {"status": "ok", "appointment_id": app_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------- Endpoint: list appointments (admin) --------
@app.get("/appointments")
async def get_appointments(admin_key: Optional[str] = None):
    # For demo, we allow listing with optional admin_key; secure this with auth in production
    return {"status": "ok", "appointments": list_appointments()}

# -------- Endpoint: chat logs (admin) --------
@app.get("/chatlogs")
async def get_chatlogs(limit: int = 50):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, user_name, user_email, timestamp, query, response, mode FROM chat_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    keys = ["id", "user_name", "user_email", "timestamp", "query", "response", "mode"]
    return {"status": "ok", "logs": [dict(zip(keys, r)) for r in rows]}

# -------- Endpoint: generate PDF of last N chat entries for a user --------
@app.post("/chat_summary_pdf")
async def chat_summary_pdf(user_email: str = Form(...), user_name: str = Form("Guest"), limit: int = Form(50)):
    # fetch last N entries for user_email
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT timestamp, query, response FROM chat_logs WHERE user_email = ? ORDER BY timestamp DESC LIMIT ?", (user_email, limit))
    rows = c.fetchall()
    conn.close()
    entries = []
    for ts, q, r in rows:
        entries.append({"timestamp": ts, "query": q, "response": r})
    # create pdf
    pdf_name = f"chat_summary_{user_email.replace('@','_')}_{int(datetime.utcnow().timestamp())}.pdf"
    pdf_path = DATA_DIR / pdf_name
    generate_chat_pdf(user_name, user_email, list(reversed(entries)), pdf_path)
    return FileResponse(str(pdf_path), filename=pdf_name, media_type='application/pdf')

# -------- Endpoint: reload CSV / rebuild KB (admin) --------
@app.post("/admin/reload_kb")
async def admin_reload_kb(admin_token: Optional[str] = Form(None)):
    # In production protect this with secure auth; for demo any token is accepted
    global KB_DF, FAQ_LIST, EMBEDDING_STORE, EMBEDDINGS_AVAILABLE
    KB_DF = load_csv(CSV_PATH)
    FAQ_LIST = [{"id": idx, "question": str(row.Question).strip(), "answer": str(row.Answer).strip()} for idx, row in KB_DF.iterrows()]
    # rebuild embeddings if available
    if USE_LANGCHAIN and OPENAI_API_KEY:
        get_embedder()
    return {"status": "ok", "message": "Knowledge base reloaded from CSV", "num_entries": len(FAQ_LIST)}

# -------- Healthcheck & info --------
@app.get("/info")
async def info():
    return {
        "service": "ConsultBot",
        "company": COMPANY_NAME,
        "salesperson_name": SALESPERSON_NAME,
        "knowledge_base_rows": len(FAQ_LIST),
        "embeddings_available": EMBEDDINGS_AVAILABLE
    }

# -------- Simple root --------
@app.get("/")
async def root():
    return {"message": "ConsultBot running. Use /consultbot with JSON POST to chat."}
