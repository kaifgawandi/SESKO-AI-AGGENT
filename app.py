import os
import sqlite3
import pdfplumber
import pytesseract
import requests
import markdown
import cv2
import numpy as np
import random 
import pytz 
import shutil
from datetime import datetime 
from PIL import Image
from bs4 import BeautifulSoup
from googleapiclient.discovery import build

# --- FASTAPI IMPORTS ---
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from pydantic import BaseModel

app = FastAPI(title="SESKO AI")

# --- SECURITY & SESSIONS ---
app.add_middleware(SessionMiddleware, secret_key="sescreate key")
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- API KEYS ---
GOOGLE_API_KEY = "???" 
SEARCH_ENGINE_ID = "???"
OLLAMA_API_URL = "???"
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- DATABASE SETUP ---
def get_db_connection():
    conn = sqlite3.connect('chat_history.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db_connection() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users 
                        (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password TEXT NOT NULL)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS sessions 
                        (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, title TEXT, is_pinned INTEGER DEFAULT 0, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(user_id) REFERENCES users(id))''')
        conn.execute('''CREATE TABLE IF NOT EXISTS history 
                        (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER, user_text TEXT, bot_text TEXT, file_path TEXT, FOREIGN KEY(session_id) REFERENCES sessions(id))''')
init_db()

class AuthData(BaseModel):
    username: str
    password: str

class RenameData(BaseModel):
    title: str

# --- AUTH ROUTES ---
@app.post('/register')
def register(data: AuthData):
    if not data.username or not data.password: 
        return JSONResponse(status_code=400, content={"error": "Missing data"})
    hashed_pw = generate_password_hash(data.password)
    try:
        with get_db_connection() as conn:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (data.username, hashed_pw))
            conn.commit()
        return {"success": True}
    except sqlite3.IntegrityError:
        return JSONResponse(status_code=400, content={"error": "Username already taken"})

@app.post('/login')
def login(request: Request, data: AuthData):
    user = get_db_connection().execute("SELECT * FROM users WHERE username = ?", (data.username,)).fetchone()
    if user and check_password_hash(user['password'], data.password):
        request.session['user_id'] = user['id']
        request.session['username'] = user['username']
        return {"success": True, "username": user['username']}
    return JSONResponse(status_code=401, content={"error": "Invalid credentials"})

@app.post('/logout')
def logout(request: Request):
    request.session.clear()
    return {"success": True}

@app.get('/check_auth')
def check_auth(request: Request):
    if 'user_id' in request.session: 
        return {"logged_in": True, "username": request.session['username']}
    return {"logged_in": False}

# --- SESSION ROUTES ---
@app.post('/new_chat')
def new_chat(request: Request):
    if 'user_id' not in request.session: 
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    with get_db_connection() as conn:
        cursor = conn.execute("INSERT INTO sessions (user_id, title) VALUES (?, ?)", (request.session['user_id'], "New Chat"))
        session_id = cursor.lastrowid
        conn.commit()
    return {"session_id": session_id}

@app.get('/get_sessions')
def get_sessions(request: Request):
    if 'user_id' not in request.session: return []
    with get_db_connection() as conn:
        sessions = conn.execute("SELECT * FROM sessions WHERE user_id = ? ORDER BY id DESC", (request.session['user_id'],)).fetchall()
    return [dict(row) for row in sessions]

@app.get('/get_chat/{session_id}')
def get_chat(session_id: int):
    with get_db_connection() as conn:
        messages = conn.execute("SELECT * FROM history WHERE session_id = ?", (session_id,)).fetchall()
    formatted = []
    for row in messages:
        msg = dict(row)
        if msg['bot_text']:
            msg['bot_text'] = markdown.markdown(msg['bot_text'], extensions=['fenced_code', 'tables', 'nl2br'])
        formatted.append(msg)
    return formatted

# --- FILE PROCESSING (PDFs & Images) ---
def process_file(path):
    """Reads text from PDFs and Images exactly like your old code"""
    try:
        # 1. Image Reading (Tesseract)
        if path.lower().endswith(('.png','.jpg','.jpeg')):
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            Image.fromarray(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]).save("temp.png")
            return f"IMAGE TEXT EXTRACTED: {pytesseract.image_to_string('temp.png')[:1500]}"
            
        # 2. PDF Reading (pdfplumber)
        if path.lower().endswith('.pdf'):
            with pdfplumber.open(path) as pdf: 
                return f"PDF CONTENT: {' '.join([p.extract_text() or '' for p in pdf.pages])[:2000]}"
    except Exception as e: 
        print(f"File processing error: {e}")
        return None

# --- INTELLIGENT BRAIN ---
def ask_local_llama(user_text, session_id=1, context=""):
    try:
        with get_db_connection() as conn:
            rows = conn.execute("SELECT user_text, bot_text FROM history WHERE session_id = ? ORDER BY id ASC", (session_id,)).fetchall()
    except: rows = []
    
    IST = pytz.timezone('Asia/Kolkata')
    date_str = datetime.now(IST).strftime("%A, %B %d, %Y at %I:%M %p")

    system_prompt = f"You are SESKO, a helpful AI assistant. Current Date and Time in India: {date_str}. When providing links, YOU MUST use Markdown format: [Title](URL). Answer briefly and professionally."
    
    # Inject File/Web Context here
    if context: system_prompt += f"\n\nCONTEXT DATA (From File/Web):\n{context}\n\nPlease analyze and answer based on the context above."

    messages = [{"role": "system", "content": system_prompt}]
    for row in rows:
        if row['user_text']: messages.append({"role": "user", "content": row['user_text']})
        if row['bot_text']: messages.append({"role": "assistant", "content": row['bot_text']})
    messages.append({"role": "user", "content": user_text})

    try:
        response = requests.post(OLLAMA_API_URL, json={"model": "llama3.2", "messages": messages, "stream": False})
        return response.json().get("message", {}).get("content", "Silent.") if response.status_code == 200 else "Error."
    except Exception as e: return f"Connection Error: {e}"

# --- GOOGLE SEARCH ---
def google_search(query, session_id=1):
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        res = service.cse().list(q=query, cx=SEARCH_ENGINE_ID, num=3, gl='in').execute()
        items = res.get('items', [])
        if not items: return "I couldn't find any real-time updates on that."
        search_context = f"REAL-TIME SEARCH RESULTS for '{query}':\n"
        for item in items: search_context += f"- Source: {item['title']}\n  Snippet: {item['snippet']}\n\n"
        prompt = f"User Question: '{query}'\n\nHere is the latest real-time information:\n{search_context}\nINSTRUCTIONS: Answer using ONLY the real-time information above."
        return ask_local_llama(prompt, session_id=session_id)
    except Exception as e: return f"Search Error: {str(e)}"

def deep_research(query, session_id=1):
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        queries = [query, f"{query} statistics data"]
        aggregated_context = f"### 🌍 DEEP RESEARCH REPORT: {query.upper()}\n\n"
        for q in queries:
            res = service.cse().list(q=q, cx=SEARCH_ENGINE_ID, num=2, gl='in').execute()
            items = res.get('items', [])
            if items:
                for item in items: aggregated_context += f"- **{item['title']}**: {item['snippet']}\n"
        prompt = f"You are an advanced Research AI. Write a 'Deep Research Report' on: '{query}'.\nData:\n{aggregated_context}\n\n**RULES:** 1. Executive Summary 2. Key Findings 3. Deep Analysis."
        return ask_local_llama(prompt, session_id=session_id)
    except Exception as e: return f"Deep Research Error: {str(e)}"

def generate_image(prompt):
    seed = random.randint(1, 9999)
    url = f"https://loremflickr.com/1024/768/{prompt.replace(' ', ',')}?lock={seed}"
    return f"### Rinz Mode Creation\n![Generated]({url})"

def scrape_website(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        soup = BeautifulSoup(requests.get(url, headers=headers, timeout=10).text, 'html.parser')
        for s in soup(["script", "style"]): s.extract()
        return f"WEB CONTENT ({url}):\n{soup.get_text()[:2000]}"
    except Exception as e: return str(e)

# --- SIDEBAR ACTIONS ---
@app.post('/delete_chat/{session_id}')
def delete_chat(session_id: int):
    with get_db_connection() as conn:
        conn.execute("DELETE FROM history WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        conn.commit()
    return {"status": "success"}

@app.post('/rename_chat/{session_id}')
def rename_chat(session_id: int, data: RenameData):
    with get_db_connection() as conn:
        conn.execute("UPDATE sessions SET title=? WHERE id=?", (data.title, session_id))
        conn.commit()
    return {"status": "success"}

@app.post('/toggle_pin/{session_id}')
def toggle_pin(session_id: int):
    with get_db_connection() as conn:
        curr = conn.execute("SELECT is_pinned FROM sessions WHERE id=?", (session_id,)).fetchone()[0]
        conn.execute("UPDATE sessions SET is_pinned=? WHERE id=?", (0 if curr else 1, session_id))
        conn.commit()
    return {"status": "success"}

# --- MAIN CHAT ROUTE (Fixed for Files) ---
@app.post('/chat')
async def chat(
    message: str = Form(""), 
    mode: str = Form("chat"), 
    session_id: int = Form(1), 
    file: UploadFile = File(None)
):
    user_text = message.strip()
    fpath, fctx = None, None
    
    # Handle File Upload & Processing correctly in FastAPI
    if file and file.filename:
        fpath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        with open(fpath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Call your restored file reader!
        fctx = process_file(fpath)

    if mode == 'image': resp = generate_image(user_text)
    elif mode == 'deep': resp = deep_research(user_text, session_id=session_id)
    elif mode == 'search': resp = google_search(user_text, session_id)
    else:
        triggers = ["news", "latest", "price", "weather", "score", "match", "live", "current", "stock", "who won", "buy", "cost", "amazon", "flipkart", "table", "standings", "league", "stats", "today", "tomorrow", "yesterday"]
        if any(t in user_text.lower() for t in triggers): resp = google_search(user_text, session_id)
        elif "http" in user_text: 
            url = [w for w in user_text.split() if w.startswith('http')][0]
            resp = ask_local_llama(f"Analyze: {user_text}", session_id, scrape_website(url))
        
        # If a file was uploaded, pass its extracted text to the brain
        elif fctx: 
            resp = ask_local_llama(user_text, session_id, context=fctx)
        else: 
            resp = ask_local_llama(user_text, session_id)

    with get_db_connection() as conn:
        conn.execute("UPDATE sessions SET title=? WHERE id=? AND title='New Chat'", (user_text[:25]+"...", session_id))
        conn.execute("INSERT INTO history (session_id, user_text, bot_text, file_path) VALUES (?,?,?,?)", (session_id, user_text, resp, fpath))
        conn.commit()

    return {"reply": markdown.markdown(resp, extensions=['fenced_code', 'tables', 'nl2br'])}

# --- SERVE HTML & LOGO ---
@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get('/sesko_logo.jpg')
def get_logo():
    return FileResponse('sesko_logo.jpg')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)

