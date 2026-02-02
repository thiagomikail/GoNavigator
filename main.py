import os
import json
import logging
import uuid
import time
import random
import gc
from typing import Dict
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import requests
import uvicorn
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import base64

import chromadb
from pypdf import PdfReader
from io import BytesIO

# APScheduler for scheduled reminders
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Load environment variables
load_dotenv()

# Configuration
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "secure_verify_token")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash-lite"
GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"  # TTS-specific model

# Persistent Data Directory (for Render disk or local storage)
# Set DATA_DIR env var to a Render persistent disk path (e.g., /var/data)
DATA_DIR = os.getenv("DATA_DIR", ".")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
print(f"[STARTUP] Data directory: {DATA_DIR}")

# Database Setup with persistent path
CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
except AttributeError:
    from chromadb.config import Settings
    chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DB_PATH))

collection = chroma_client.get_or_create_collection(name="trip_knowledge")

# Simple Session Store (persistent path)
SESSION_FILE = os.path.join(DATA_DIR, "trip_sessions.json")

# Message deduplication (file-based for persistence across restarts)
PROCESSED_MESSAGES_FILE = os.path.join(DATA_DIR, "processed_messages.json")
MESSAGE_EXPIRY_SECONDS = 3600  # Keep message IDs for 1 hour

def load_processed_messages() -> dict:
    """Load processed message IDs from file."""
    if os.path.exists(PROCESSED_MESSAGES_FILE):
        try:
            with open(PROCESSED_MESSAGES_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_processed_message(message_id: str, timestamp: float):
    """Save a processed message ID to file."""
    messages = load_processed_messages()
    current_time = time.time()
    
    # Clean up expired messages
    messages = {mid: ts for mid, ts in messages.items() if current_time - ts < MESSAGE_EXPIRY_SECONDS}
    
    # Add new message
    messages[message_id] = timestamp
    
    with open(PROCESSED_MESSAGES_FILE, "w") as f:
        json.dump(messages, f)

def is_duplicate_message(message_id: str) -> bool:
    """Check if message was already processed."""
    messages = load_processed_messages()
    return message_id in messages

def load_sessions() -> Dict[str, str]:
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, "r") as f:
            return json.load(f)
    return {}

def save_session(phone: str, trip_id: str):
    sessions = load_sessions()
    sessions[phone] = trip_id.upper()
    with open(SESSION_FILE, "w") as f:
        json.dump(sessions, f)

# --- User Management (Roles: superuser, employee) ---
USERS_FILE = "users.json"
import hashlib

def hash_password(password: str) -> str:
    """Simple SHA256 hash for passwords."""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users() -> Dict[str, Dict]:
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_users(users: Dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def authenticate_user(email: str, password: str) -> Dict:
    """Authenticate user and return user data with role, or None if invalid."""
    users = load_users()
    
    # Fallback: check legacy ADMIN_PASSWORD env var for superuser
    admin_pass = os.getenv("ADMIN_PASSWORD", "secret123")
    if password == admin_pass:
        return {"email": "admin", "role": "superuser"}
    
    # Check registered users
    for user_id, user_data in users.items():
        if user_data.get("email") == email:
            if user_data.get("password_hash") == hash_password(password):
                return {"email": email, "role": user_data.get("role", "employee")}
    return None

def is_superuser(auth_result: Dict) -> bool:
    """Check if authenticated user is superuser."""
    return auth_result and auth_result.get("role") == "superuser"

# --- Settings Management ---
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")

def load_settings() -> Dict:
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {
        "suggestion_links": {
            "ideias_viagem": "https://placeholder.com/ideias",
            "miles_tips": "https://placeholder.com/milhas"
        }
    }

def save_settings(settings: Dict):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)

# --- LLM Configuration & Retry Logic ---
LLM_MAX_RETRIES = 3
LLM_BASE_DELAY = 1.0  # seconds
LLM_N_RESULTS = 3  # Reduced from 5 for token efficiency

def call_gemini_with_retry(prompt: str, max_retries: int = LLM_MAX_RETRIES) -> str:
    """Call Gemini API with exponential backoff retry on 429/5xx errors."""
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, json=payload, timeout=30)
            
            if response.status_code == 429:
                # Rate limited - exponential backoff with jitter
                delay = LLM_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"[GEMINI] Rate limited (429). Retry {attempt+1}/{max_retries} in {delay:.1f}s")
                time.sleep(delay)
                continue
            
            if response.status_code >= 500:
                # Server error - retry
                delay = LLM_BASE_DELAY * (2 ** attempt)
                logger.warning(f"[GEMINI] Server error ({response.status_code}). Retry {attempt+1}/{max_retries} in {delay:.1f}s")
                time.sleep(delay)
                continue
            
            response.raise_for_status()
            return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            
        except requests.exceptions.Timeout:
            logger.warning(f"[GEMINI] Timeout. Retry {attempt+1}/{max_retries}")
            continue
        except Exception as e:
            logger.error(f"[GEMINI] Error: {e}")
            if attempt == max_retries - 1:
                raise
    
    raise Exception("Gemini API failed after max retries")

# --- Trip Management ---
TRIPS_FILE = os.path.join(DATA_DIR, "trips.json")

def load_trips() -> Dict[str, Dict]:
    if os.path.exists(TRIPS_FILE):
        try:
            with open(TRIPS_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_trip(trip_id: str, settings: Dict = None):
    trips = load_trips()
    if trip_id not in trips:
        trips[trip_id] = {
            "created_at": time.time(),
            "voice_enabled": True,
            "ai_enabled": True
        }
    if settings:
        trips[trip_id].update(settings)
    
    with open(TRIPS_FILE, "w") as f:
        json.dump(trips, f, indent=2)

def delete_trip_data(trip_id: str):
    # 1. Provide trip from trips.json
    trips = load_trips()
    if trip_id in trips:
        del trips[trip_id]
        with open(TRIPS_FILE, "w") as f:
             json.dump(trips, f)
    
    # 2. Delete from ChromaDB
    try:
        collection.delete(where={"trip_id": trip_id})
    except Exception as e:
        logger.error(f"Error deleting from Chroma: {e}")

    # 3. Clear from sessions
    sessions = load_sessions()
    # Find keys (phones) that have this trip_id
    phones_to_remove = [phone for phone, tid in sessions.items() if tid == trip_id]
    for phone in phones_to_remove:
        del sessions[phone]
    with open(SESSION_FILE, "w") as f:
        json.dump(sessions, f)

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mount Static Files
if not os.path.exists("static"):
    os.makedirs("static")
if not os.path.exists("static/audio"):
    os.makedirs("static/audio")

app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Helper: Text to Speech using Gemini ---
def generate_audio(text: str) -> str:
    """Generates MP3 audio from text using Gemini TTS and returns the relative filename."""
    try:
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_TTS_MODEL}:generateContent?key={GOOGLE_API_KEY}"
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": f"Fale em português brasileiro: {text}"
                }]
            }],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": "Kore"
                        }
                    }
                }
            }
        }
        
        response = requests.post(api_url, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        if "candidates" in result and result["candidates"]:
            parts = result["candidates"][0].get("content", {}).get("parts", [])
            for part in parts:
                if "inlineData" in part:
                    pcm_data = base64.b64decode(part["inlineData"]["data"])
                    
                    import subprocess
                    import tempfile
                    
                    filename = f"audio_{uuid.uuid4()}.mp3"
                    filepath = os.path.join("static/audio", filename)
                    
                    with tempfile.NamedTemporaryFile(suffix='.pcm', delete=False) as tmp:
                        tmp.write(pcm_data)
                        tmp_path = tmp.name
                    
                    try:
                        subprocess.run([
                            'ffmpeg', '-y',
                            '-f', 's16le', '-ar', '24000', '-ac', '1',
                            '-i', tmp_path, '-b:a', '64k', filepath
                        ], check=True, capture_output=True)
                        return filename
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
        
        logger.warning("No audio data in Gemini response")
        return None
        
    except Exception as e:
        logger.error(f"Gemini TTS Error: {e}", exc_info=True)
        return None

# --- Ingestion Logic (Memory-Optimized) ---
CHUNK_SIZE = 2000  # Larger chunks capture more context
CHUNK_OVERLAP = 200  # More overlap to avoid missing info at boundaries
BATCH_SIZE = 20  # Insert chunks in batches to reduce memory

def process_pdf_upload(file_stream, trip_id):
    """Memory-efficient PDF processing with lazy page-by-page extraction."""
    logger.info(f"[STAGE 1: PDF PARSING] Starting for trip {trip_id}")
    
    reader = PdfReader(file_stream)
    total_pages = len(reader.pages)
    
    # Process page-by-page to avoid holding entire text in memory
    chunk_buffer = []
    chunk_index = 0
    leftover_text = ""
    
    for page_num in range(total_pages):
        # Extract single page text
        page_text = reader.pages[page_num].extract_text() or ""
        logger.info(f"[STAGE 1: PDF PARSING] Page {page_num+1}/{total_pages}: {len(page_text)} chars")
        
        # Combine with leftover from previous page
        current_text = leftover_text + page_text
        
        # Create chunks from current text
        pos = 0
        while pos + CHUNK_SIZE <= len(current_text):
            chunk = current_text[pos:pos + CHUNK_SIZE]
            chunk_buffer.append({
                "id": f"{trip_id}_{chunk_index}",
                "text": chunk,
                "meta": {"source": "upload", "trip_id": trip_id}
            })
            chunk_index += 1
            pos += CHUNK_SIZE - CHUNK_OVERLAP
            
            # Insert in batches to limit memory
            if len(chunk_buffer) >= BATCH_SIZE:
                _insert_chunk_batch(chunk_buffer, trip_id)
                chunk_buffer = []
        
        # Keep leftover for next page
        leftover_text = current_text[pos:] if pos < len(current_text) else ""
        
        # Force garbage collection every 50 pages
        if page_num > 0 and page_num % 50 == 0:
            gc.collect()
    
    # Handle remaining text as final chunk
    if leftover_text.strip():
        chunk_buffer.append({
            "id": f"{trip_id}_{chunk_index}",
            "text": leftover_text,
            "meta": {"source": "upload", "trip_id": trip_id}
        })
    
    # Insert remaining chunks
    if chunk_buffer:
        _insert_chunk_batch(chunk_buffer, trip_id)
    
    logger.info(f"[STAGE 2: CHROMADB INSERT] Complete. {chunk_index + 1} chunks indexed for trip {trip_id}")
    
    # Cleanup
    del reader
    gc.collect()
    
    try:
        chroma_client.persist()
        logger.info(f"[STAGE 2: CHROMADB INSERT] Database persisted")
    except AttributeError:
        pass

def _insert_chunk_batch(chunks: list, trip_id: str):
    """Insert a batch of chunks into ChromaDB."""
    if not chunks:
        return
    
    ids = [c["id"] for c in chunks]
    docs = [c["text"] for c in chunks]
    metas = [c["meta"] for c in chunks]
    
    collection.add(documents=docs, ids=ids, metadatas=metas)
    logger.info(f"[STAGE 2: CHROMADB INSERT] Batch inserted: {len(chunks)} chunks")

@app.post("/upload_trip")
async def upload_trip(
    trip_id: str = Form(...),
    file: UploadFile = File(...),
    admin_key: str = Form(...)
):
    if admin_key != os.getenv("ADMIN_PASSWORD", "secret123"):
        raise HTTPException(status_code=401, detail="Senha incorreta")
        
    try:
        content = await file.read()
        file_stream = BytesIO(content)
        process_pdf_upload(file_stream, trip_id.upper())
        save_trip(trip_id.upper()) # Save metadata
        return {"status": "success", "trip_id": trip_id, "message": "Trip created and PDF indexed."}
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Admin API ---

# Authentication
@app.post("/api/auth/login")
async def login(request: Request):
    data = await request.json()
    email = data.get("email", "")
    password = data.get("password", "")
    
    user = authenticate_user(email, password)
    if user:
        return {"status": "success", "user": user}
    raise HTTPException(status_code=401, detail="Invalid credentials")

# User Management (Superuser only)
@app.get("/api/users")
async def list_users(admin_key: str):
    user = authenticate_user("", admin_key)
    if not is_superuser(user):
        raise HTTPException(status_code=403, detail="Superuser access required")
    return load_users()

@app.post("/api/users")
async def create_user(request: Request, admin_key: str):
    user = authenticate_user("", admin_key)
    if not is_superuser(user):
        raise HTTPException(status_code=403, detail="Superuser access required")
    
    data = await request.json()
    email = data.get("email")
    password = data.get("password")
    role = data.get("role", "employee")
    
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")
    
    users = load_users()
    user_id = email.split("@")[0]
    users[user_id] = {
        "email": email,
        "password_hash": hash_password(password),
        "role": role,
        "created_at": time.time()
    }
    save_users(users)
    return {"status": "created", "email": email, "role": role}

@app.delete("/api/users/{user_id}")
async def delete_user(user_id: str, admin_key: str):
    user = authenticate_user("", admin_key)
    if not is_superuser(user):
        raise HTTPException(status_code=403, detail="Superuser access required")
    
    users = load_users()
    if user_id in users:
        del users[user_id]
        save_users(users)
        return {"status": "deleted", "user_id": user_id}
    raise HTTPException(status_code=404, detail="User not found")

# Settings Management
@app.get("/api/settings")
async def get_settings(admin_key: str):
    user = authenticate_user("", admin_key)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return load_settings()

@app.put("/api/settings")
async def update_settings(request: Request, admin_key: str):
    user = authenticate_user("", admin_key)
    if not is_superuser(user):
        raise HTTPException(status_code=403, detail="Superuser access required")
    
    data = await request.json()
    settings = load_settings()
    settings.update(data)
    save_settings(settings)
    return {"status": "updated", "settings": settings}

# Database Stats
@app.get("/api/database/stats")
async def get_database_stats(admin_key: str):
    user = authenticate_user("", admin_key)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    trips = load_trips()
    stats = {"total_trips": len(trips), "trips": {}, "total_chunks": 0}
    
    # Count chunks per trip
    try:
        all_results = collection.get(include=["metadatas"])
        trip_chunks = {}
        for meta in all_results.get("metadatas", []):
            tid = meta.get("trip_id", "unknown")
            trip_chunks[tid] = trip_chunks.get(tid, 0) + 1
        
        stats["trips"] = trip_chunks
        stats["total_chunks"] = sum(trip_chunks.values())
    except Exception as e:
        logger.error(f"Error getting DB stats: {e}")
    
    # Get DB size
    db_path = "chroma_db"
    if os.path.exists(db_path):
        total_size = sum(os.path.getsize(os.path.join(dp, f)) 
                        for dp, dn, fn in os.walk(db_path) for f in fn)
        stats["db_size_mb"] = round(total_size / (1024 * 1024), 2)
    
    return stats

# Trip Management
@app.get("/api/trips")
async def list_trips_api(admin_key: str):
    user = authenticate_user("", admin_key)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return load_trips()

@app.delete("/api/trips/{trip_id}")
async def delete_trip(trip_id: str, admin_key: str):
    user = authenticate_user("", admin_key)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    delete_trip_data(trip_id)
    return {"status": "deleted", "trip_id": trip_id}

@app.post("/api/trips/{trip_id}/settings")
async def update_trip_settings(trip_id: str, request: Request, admin_key: str):
    user = authenticate_user("", admin_key)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    settings = await request.json()
    save_trip(trip_id, settings)
    return {"status": "updated", "trip_id": trip_id, "settings": settings}

# Timeline Generator
@app.get("/api/trips/{trip_id}/timeline")
async def get_timeline_page(trip_id: str):
    """Render timeline HTML page for a trip."""
    trips = load_trips()
    if trip_id.upper() not in trips:
        raise HTTPException(status_code=404, detail="Trip not found")
    
    # Read and render template
    with open("static/timeline.html", "r", encoding="utf-8") as f:
        html = f.read()
    
    html = html.replace("{{TRIP_ID}}", trip_id.upper())
    return PlainTextResponse(content=html, media_type="text/html")

@app.get("/api/trips/{trip_id}/timeline/data")
async def get_timeline_data(trip_id: str):
    """Extract and return timeline events as JSON with title and dates."""
    trips = load_trips()
    if trip_id.upper() not in trips:
        raise HTTPException(status_code=404, detail="Trip not found")
    
    try:
        # Get all chunks for this trip
        results = collection.query(
            query_texts=["roteiro completo programacao atividades destino"],
            n_results=15,
            where={"trip_id": trip_id.upper()}
        )
        
        if not results['documents'][0]:
            return {"title": trip_id.upper(), "dates": "", "events": []}
        
        context = "\n".join(results['documents'][0])
        
        # Use Gemini to extract structured events with title info
        prompt = f"""Extraia as informacoes do roteiro abaixo e retorne como JSON.

Formato esperado:
{{
  "title": "DESTINOS DA VIAGEM",
  "dates": "DD/MM/YYYY - DD/MM/YYYY",
  "events": [
    {{"day": "DAY 1", "location": "Nome do Local", "description": "Breve descricao"}},
    ...
  ]
}}

Regras:
- title: Destino(s) principal(is) em letras maiusculas (ex: "LONDRES", "PARIS E ROMA")
- dates: Data de inicio e fim da viagem
- events: Lista de dias/atividades em ordem cronologica
- day: Use formato "DAY 1", "DAY 2", etc
- location: Nome do local/cidade principal do dia
- description: Breve descricao da atividade (opcional)
- Responda APENAS com o JSON

Roteiro:
{context}

JSON:"""
        
        response_text = call_gemini_with_retry(prompt)
        
        # Parse JSON from response
        import re
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            data = json.loads(json_match.group())
            return data
        
        return {"title": trip_id.upper(), "dates": "", "events": []}
        
    except Exception as e:
        logger.error(f"Timeline generation error: {e}")
        return {"title": trip_id.upper(), "dates": "", "events": []}

@app.post("/api/trips/{trip_id}/append")
async def append_trip_document(
    trip_id: str,
    file: UploadFile = File(...),
    admin_key: str = Form(...)
):
    user = authenticate_user("", admin_key)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    trips = load_trips()
    if trip_id.upper() not in trips:
        raise HTTPException(status_code=404, detail="Trip not found")
    
    try:
        content = await file.read()
        file_stream = BytesIO(content)
        process_pdf_upload(file_stream, trip_id.upper())
        return {"status": "success", "trip_id": trip_id, "message": "Document appended to trip."}
    except Exception as e:
        logger.error(f"Append failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.api_route("/", methods=["GET", "HEAD"])
async def health_check():
    return "GoNavigator Online. Go to /static/admin.html for Admin."

@app.get("/webhook")
async def verify_webhook(request: Request):
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    logger.info(f"WEBHOOK VERIFY: mode={mode} token={token} challenge={challenge}")

    if mode and token:
        if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
            logger.info("WEBHOOK VERIFICATION SUCCESSFUL")
            return PlainTextResponse(content=str(challenge) if challenge else "", status_code=200)
        else:
             logger.warning(f"WEBHOOK VERIFICATION FAILED: Expected {WHATSAPP_VERIFY_TOKEN}, got {token}")
    
    raise HTTPException(status_code=403, detail="Verification failed")

@app.post("/webhook")
async def receive_message(request: Request, background_tasks: BackgroundTasks):
    """Fast webhook handler - returns 200 OK immediately, processes in background."""
    data = await request.json()
    logger.info(f"[WEBHOOK] Payload received, queuing for background processing")
    
    # Return 200 immediately to prevent WhatsApp retry loops
    # WhatsApp will retry if we don't respond within 10-20 seconds
    background_tasks.add_task(process_webhook_message, data, str(request.base_url).rstrip("/"))
    return {"status": "accepted"}

def process_webhook_message(data: dict, base_url: str):
    """Background task to process incoming WhatsApp messages."""
    try:
        entry = data.get('entry', [])[0]
        changes = entry.get('changes', [])[0]
        value = changes.get('value', {})
        
        if 'messages' not in value:
            return
            
        message = value['messages'][0]
        message_id = message.get('id', '')
        from_number = message['from']
        msg_body = message.get('text', {}).get('body', '').strip()
        
        # Deduplicate: skip if already processed
        if is_duplicate_message(message_id):
            logger.info(f"[DEDUP] Skipping duplicate message: {message_id}")
            return
        
        # Track this message ID
        save_processed_message(message_id, time.time())
        
        logger.info(f"[WEBHOOK] Processing: From {from_number}: {msg_body}")
        
        sessions = load_sessions()
        user_trip_id = sessions.get(from_number)
        
        if not user_trip_id:
            # Handle menu responses (1-4) or trip codes
            if msg_body in ['1', '2', '3', '4']:
                handle_menu_response(from_number, msg_body)
            elif len(msg_body) < 15 and msg_body.replace(" ", "").isalnum():
                # Validate trip code
                trip_code = msg_body.upper().strip()
                trips = load_trips()
                if trip_code in trips:
                    save_session(from_number, trip_code)
                    send_whatsapp_message(from_number, f"✅ Código {trip_code} registrado! Pode perguntar sobre sua viagem.")
                else:
                    send_whatsapp_message(from_number, f"❌ Código '{trip_code}' não encontrado. Verifique ou escolha uma opção abaixo.")
                    send_suggestion_menu(from_number)
            else:
                # First contact - send suggestion menu
                send_suggestion_menu(from_number)
        else:
            handle_qa(from_number, user_trip_id, msg_body, base_url)
                
    except Exception as e:
        logger.error(f"[WEBHOOK] Background processing error: {e}", exc_info=True)

def send_suggestion_menu(phone: str):
    """Send formatted text menu with numbered options."""
    menu = """👋 Olá! Sou o assistente virtual da sua agência de viagens.

Escolha uma opção:

1️⃣ Digitar código da viagem
2️⃣ Idéias de Viagens 🌍
3️⃣ Dicas de Milhas ✈️
4️⃣ Falar com um agente

➡️ Responda com o número da opção (1, 2, 3 ou 4)"""
    send_whatsapp_message(phone, menu)

def handle_menu_response(phone: str, choice: str):
    """Handle user's menu choice."""
    settings = load_settings()
    links = settings.get("suggestion_links", {})
    
    if choice == '1':
        send_whatsapp_message(phone, "Digite o *Código da Viagem* que você recebeu da agência (ex: PARIS24, LON1):")
    
    elif choice == '2':
        url = links.get("ideias_viagem", "https://gonavigator.com/ideias")
        send_whatsapp_message(phone, f"🌍 *Idéias de Viagens*\n\nDescubra destinos incríveis para sua próxima aventura!\n\n🔗 {url}")
    
    elif choice == '3':
        url = links.get("miles_tips", "https://gonavigator.com/milhas")
        send_whatsapp_message(phone, f"✈️ *Dicas de Milhas*\n\nAprenda a economizar e viajar mais usando milhas!\n\n🔗 {url}")
    
    elif choice == '4':
        send_handoff_message(phone, "Solicitação de atendimento humano")

def handle_qa(user_phone: str, trip_id: str, query: str, base_url: str):
    # 1. Check Trip Settings (AI Enabled?)
    trips = load_trips()
    trip_config = trips.get(trip_id, {"ai_enabled": True, "voice_enabled": True})
    
    if not trip_config.get("ai_enabled", True):
        send_whatsapp_message(user_phone, "O assistente virtual está temporariamente desativado para esta viagem.")
        return

    logger.info(f"[STAGE 3: CHROMADB QUERY] Searching for trip {trip_id}, query: '{query[:50]}...'")
    query_start = time.time()
    
    results = collection.query(
        query_texts=[query],
        n_results=8,  # More chunks for better coverage of specific mentions
        where={"trip_id": trip_id}
    )

    num_docs = len(results['documents'][0]) if results['documents'] else 0
    query_time = time.time() - query_start
    logger.info(f"[STAGE 3: CHROMADB QUERY] Found {num_docs} chunks in {query_time:.2f}s")
    
    if not results['documents'][0]:
        logger.warning(f"[STAGE 3: CHROMADB QUERY] FAILED - No documents for {trip_id}")
        send_handoff_message(user_phone, query)
        return

    # Log each chunk briefly
    for i, doc in enumerate(results['documents'][0]):
        logger.info(f"[STAGE 3: CHROMADB QUERY] Chunk {i}: {len(doc)} chars, preview: '{doc[:80]}...'")

    context = "\n".join(results['documents'][0])
    
    # Write debug info to file for inspection (accessible at /static/debug/)
    debug_dir = "static/debug"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    debug_file = os.path.join(debug_dir, f"debug_{timestamp}_{trip_id}.txt")
    
    # Improved prompt - helpful and complete while secure
    prompt = f"""Você é um assistente de viagem amigável para a viagem {trip_id}.

**Instruções:**
- Use as informações do roteiro abaixo para responder.
- Seja DETALHADO e COMPLETO na resposta. Inclua datas, horários, endereços e dicas quando disponíveis.
- Se a pergunta pede uma lista (ex: restaurantes, passeios), liste TODOS os itens relevantes encontrados.
- Se o roteiro menciona o assunto mas não tem todos os detalhes, responda com o que você encontrar.
- APENAS diga "HANDOFF_REQUIRED" se o assunto NÃO aparecer no roteiro de forma alguma.
- Responda em português do Brasil, de forma natural e amigável.

**Roteiro da Viagem:**
{context}

**Pergunta do Viajante:** {query}

**Resposta Completa:**"""
    
    logger.info(f"[STAGE 4: GEMINI] Sending prompt ({len(prompt)} chars) to {GEMINI_MODEL}")
    gemini_start = time.time()
    
    try:
        # Use retry wrapper for 429 handling
        reply_text = call_gemini_with_retry(prompt)
        gemini_time = time.time() - gemini_start
        
        logger.info(f"[STAGE 4: GEMINI] Response received in {gemini_time:.2f}s ({len(reply_text)} chars)")
        
        # Write debug file with all info
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(f"=== DEBUG LOG ===\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Trip ID: {trip_id}\n")
            f.write(f"User Phone: {user_phone}\n")
            f.write(f"Query: {query}\n\n")
            f.write(f"=== CHUNKS FROM CHROMADB ({num_docs} chunks) ===\n")
            for i, doc in enumerate(results['documents'][0]):
                f.write(f"\n--- Chunk {i} ({len(doc)} chars) ---\n")
                f.write(doc)
                f.write("\n")
            f.write(f"\n=== TOTAL CONTEXT ({len(context)} chars) ===\n")
            f.write(context)
            f.write(f"\n\n=== FULL PROMPT ({len(prompt)} chars) ===\n")
            f.write(prompt)
            f.write(f"\n\n=== GEMINI RESPONSE ({len(reply_text)} chars) ===\n")
            f.write(reply_text)
        
        logger.info(f"[DEBUG] File written: {debug_file}")
        
        if "HANDOFF_REQUIRED" in reply_text:
            logger.info("LLM triggered Handoff.")
            send_handoff_message(user_phone, query)
        else:
            # 1. Send Text
            send_whatsapp_message(user_phone, reply_text)
            
            # 2. Generate & Send Audio (If Voice Enabled)
            if trip_config.get("voice_enabled", True):
                if not os.path.exists("static/audio"):
                    os.makedirs("static/audio")
                    
                audio_file = generate_audio(reply_text)
                if audio_file:
                    audio_url = f"{base_url}/static/audio/{audio_file}"
                    logger.info(f"Sending audio: {audio_url}")
                    send_whatsapp_audio(user_phone, audio_url)
                else:
                    logger.warning("Audio generation returned None. Check TTS logic.")
            else:
                 logger.info(f"Voice disabled for trip {trip_id}, skipping audio.")
            
    except Exception as e:
        logger.error(f"LLM/TTS Error: {e}", exc_info=True)
        send_whatsapp_message(user_phone, "Desculpe, tive um erro técnico.")

def send_handoff_message(user_phone: str, query: str):
    send_whatsapp_message(user_phone, "Não encontrei essa informação precisa no roteiro. Já notifiquei o agente responsável para te ajudar!")
    with open("concierge_log.txt", "a") as f:
        f.write(f"URGENT: User {user_phone} asked: '{query}'\n")

def send_whatsapp_message(to_number: str, text: str):
    url = f"https://graph.facebook.com/v17.0/{WHATSAPP_PHONE_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "text": {"body": text}
    }
    try:
        requests.post(url, json=payload, headers=headers)
    except Exception as e:
        logger.error(f"Failed to send message: {e}")

def send_whatsapp_audio(to_number: str, audio_url: str):
    url = f"https://graph.facebook.com/v17.0/{WHATSAPP_PHONE_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "audio",
        "audio": {"link": audio_url}
    }
    try:
        r = requests.post(url, json=payload, headers=headers)
        logger.info(f"Audio sent status: {r.status_code} {r.text}")
    except Exception as e:
        logger.error(f"Failed to send audio: {e}")

# --- Scheduled Reminders ---
scheduler = BackgroundScheduler()

def send_scheduled_reminders():
    """Send 12-hour schedule reminders to all active trip users."""
    logger.info("[SCHEDULER] Running scheduled reminder job...")
    
    try:
        trips = load_trips()
        sessions = load_sessions()
        
        today = datetime.now().strftime("%d/%m/%Y")
        
        for trip_id, trip_config in trips.items():
            if not trip_config.get("reminder_enabled", False):
                continue
            
            trip_users = [phone for phone, tid in sessions.items() if tid == trip_id]
            
            if not trip_users:
                continue
            
            logger.info(f"[SCHEDULER] Sending reminder for {trip_id} to {len(trip_users)} users")
            
            schedule_text = generate_trip_schedule(trip_id, today)
            
            if schedule_text:
                for phone in trip_users:
                    send_whatsapp_message(phone, schedule_text)
                    
    except Exception as e:
        logger.error(f"[SCHEDULER] Reminder job error: {e}", exc_info=True)

def generate_trip_schedule(trip_id: str, date: str) -> str:
    """Generate a 12-hour schedule summary using RAG."""
    try:
        results = collection.query(
            query_texts=[f"programacao agenda atividades do dia {date}"],
            n_results=5,
            where={"trip_id": trip_id}
        )
        
        if not results['documents'][0]:
            return None
        
        context = "\n".join(results['documents'][0])
        
        prompt = f"""Com base no roteiro, crie um resumo das atividades de hoje para o viajante.

Data: {date}

Formato:
📅 *Programacao de Hoje*

🕐 [horario] - [atividade]
...

Roteiro:
{context}

Resumo:"""
        
        return call_gemini_with_retry(prompt)
        
    except Exception as e:
        logger.error(f"[SCHEDULER] Schedule generation error: {e}")
        return None

def init_scheduler():
    """Initialize the reminder scheduler."""
    settings = load_settings()
    reminder_config = settings.get("reminder_defaults", {})
    reminder_times = reminder_config.get("times", ["08:00", "20:00"])
    
    scheduler.remove_all_jobs()
    
    for time_str in reminder_times:
        try:
            hour, minute = map(int, time_str.split(":"))
            trigger = CronTrigger(hour=hour, minute=minute, timezone="America/Sao_Paulo")
            scheduler.add_job(send_scheduled_reminders, trigger, id=f"reminder_{time_str}")
            logger.info(f"[SCHEDULER] Added reminder job at {time_str}")
        except Exception as e:
            logger.error(f"[SCHEDULER] Failed to add job for {time_str}: {e}")
    
    if not scheduler.running:
        scheduler.start()
        logger.info("[SCHEDULER] Started background scheduler")

@app.on_event("startup")
async def startup_event():
    init_scheduler()

@app.on_event("shutdown")
async def shutdown_event():
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("[SCHEDULER] Shutdown")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
