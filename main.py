import os
import json
import logging
import uuid
import time
from typing import Dict

import requests
import uvicorn
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import base64

import chromadb
from pypdf import PdfReader
from io import BytesIO

# Load environment variables
load_dotenv()

# Configuration
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "secure_verify_token")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash-lite"
GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"  # TTS-specific model

# Database Setup (Use ChromaDB's default local embeddings for reliability)
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
except AttributeError:
    from chromadb.config import Settings
    chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db"))

collection = chroma_client.get_or_create_collection(name="trip_knowledge")

# Simple Session Store
SESSION_FILE = "trip_sessions.json"

# Message deduplication (prevent WhatsApp retry duplicates)
processed_messages = {}  # {message_id: timestamp}
MESSAGE_EXPIRY_SECONDS = 3600  # Keep message IDs for 1 hour

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

# --- Trip Management ---
TRIPS_FILE = "trips.json"

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

# --- Ingestion Logic ---
def process_pdf_upload(file_stream, trip_id):
    logger.info(f"[STAGE 1: PDF PARSING] Starting for trip {trip_id}")
    
    reader = PdfReader(file_stream)
    text = ""
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        text += page_text + "\n"
        logger.info(f"[STAGE 1: PDF PARSING] Page {i+1}: {len(page_text)} chars extracted")
    
    logger.info(f"[STAGE 1: PDF PARSING] Complete. Total: {len(reader.pages)} pages, {len(text)} chars")
        
    chunk_size = 1000
    overlap = 100
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    
    logger.info(f"[STAGE 2: CHROMADB INSERT] Creating {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
        
    ids = [f"{trip_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": "upload", "trip_id": trip_id} for _ in chunks]
    
    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )
    
    logger.info(f"[STAGE 2: CHROMADB INSERT] Complete. {len(chunks)} chunks indexed for trip {trip_id}")
    
    try:
        chroma_client.persist()
        logger.info(f"[STAGE 2: CHROMADB INSERT] Database persisted")
    except AttributeError:
        pass 

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
@app.get("/api/trips")
async def list_trips(admin_key: str):
    if admin_key != os.getenv("ADMIN_PASSWORD", "secret123"):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return load_trips()

@app.delete("/api/trips/{trip_id}")
async def delete_trip(trip_id: str, admin_key: str):
    if admin_key != os.getenv("ADMIN_PASSWORD", "secret123"):
         raise HTTPException(status_code=401, detail="Unauthorized")
    delete_trip_data(trip_id)
    return {"status": "deleted", "trip_id": trip_id}

@app.post("/api/trips/{trip_id}/settings")
async def update_trip_settings(trip_id: str, settings: Dict, admin_key: str):
    if admin_key != os.getenv("ADMIN_PASSWORD", "secret123"):
         raise HTTPException(status_code=401, detail="Unauthorized")
    save_trip(trip_id, settings)
    return {"status": "updated", "trip_id": trip_id, "settings": settings}

@app.api_route("/", methods=["GET", "HEAD"])
async def health_check():
    return "GoNavigator Online. Go to /static/index.html for Admin."

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
async def receive_message(request: Request):
    data = await request.json()
    logger.info(f"RAW PAYLOAD RECEIVED: {json.dumps(data)}")
    
    try:
        entry = data.get('entry', [])[0]
        changes = entry.get('changes', [])[0]
        value = changes.get('value', {})
        
        if 'messages' in value:
            message = value['messages'][0]
            message_id = message.get('id', '')
            from_number = message['from']
            msg_body = message.get('text', {}).get('body', '').strip()
            
            # Deduplicate: skip if already processed
            current_time = time.time()
            if message_id in processed_messages:
                logger.info(f"Skipping duplicate message: {message_id}")
                return {"status": "duplicate"}
            
            # Track this message ID
            processed_messages[message_id] = current_time
            
            # Cleanup old message IDs (prevent memory leak)
            expired = [mid for mid, ts in processed_messages.items() if current_time - ts > MESSAGE_EXPIRY_SECONDS]
            for mid in expired:
                del processed_messages[mid]
            
            logger.info(f"From {from_number}: {msg_body}")
            
            sessions = load_sessions()
            user_trip_id = sessions.get(from_number)
            
            if not user_trip_id:
                if len(msg_body) < 15 and msg_body.isalnum():
                     trip_code = msg_body.upper()
                     # VALIDATE TRIP EXISTS
                     trips = load_trips()
                     if trip_code in trips:
                         save_session(from_number, trip_code)
                         send_whatsapp_message(from_number, f"Bem-vindo! Código {trip_code} registrado. Pode perguntar sobre sua viagem.")
                     else:
                         send_whatsapp_message(from_number, f"Código {trip_code} não encontrado. Por favor, verifique ou contate sua agência.")
                else:
                    send_whatsapp_message(from_number, "Olá! Sou o assistente virtual da sua agência. Por favor, digite o *Código da Viagem* (ex: PARIS24) para começar.")
            else:
                base_url = str(request.base_url).rstrip("/")
                handle_qa(from_number, user_trip_id, msg_body, base_url)
                
    except Exception as e:
        logger.error(f"Error processing webhook: {e}", exc_info=True)
        
    return {"status": "processed"}

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
        n_results=5,
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
    
    prompt = (
        f"Você é um guia de viagem para a viagem código {trip_id}. "
        f"RESPONDA APENAS À PERGUNTA ESPECÍFICA do viajante. "
        f"NÃO mencione informações sobre outros assuntos que não foram perguntados. "
        f"Se a pergunta é sobre passeios, fale SOMENTE de passeios. Se é sobre restaurantes, fale SOMENTE de restaurantes. "
        f"Use APENAS as informações do contexto abaixo. "
        f"Se a informação não estiver no contexto, DIGA: 'HANDOFF_REQUIRED'. "
        f"Responda em Português do Brasil, de forma completa.\n\n"
        f"Contexto do Roteiro:\n{context}\n\n"
        f"Pergunta do Viajante: {query}"
    )
    
    logger.info(f"[STAGE 4: GEMINI] Sending prompt ({len(prompt)} chars) to {GEMINI_MODEL}")
    gemini_start = time.time()
    
    try:
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GOOGLE_API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()
        
        reply_text = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        gemini_time = time.time() - gemini_start
        
        logger.info(f"[STAGE 4: GEMINI] Response received in {gemini_time:.2f}s ({len(reply_text)} chars)")
        logger.info(f"[STAGE 4: GEMINI] Response preview: '{reply_text[:100]}...'")
        
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
