import os
import json
import logging
from typing import Dict

import requests
import uvicorn
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

import google.generativeai as genai
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

# Initialize Clients
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Database Setup
# Database Setup
try:
    # New API (ChromaDB 0.4.x+)
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
except AttributeError:
    # Old API (ChromaDB 0.3.x) - Fallback for Python 3.14 local environments
    from chromadb.config import Settings
    chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db"))

collection = chroma_client.get_or_create_collection(name="trip_knowledge")

# Simple Session Store (In production, use Redis or SQL)
# Format: { "phone_number": "TRIP_ID" }
# We persist this to a JSON file so it survives restarts in this simple demo
SESSION_FILE = "trip_sessions.json"

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

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mount Static Files (Admin UI)
# Create directory if not exists
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/admin", StaticFiles(directory="static", html=True), name="static")

# --- Ingestion Logic (Moved from ingest.py to Server) ---
def process_pdf_upload(file_stream, trip_id):
    # 1. Read PDF
    reader = PdfReader(file_stream)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
        
    # 2. Chunk
    chunk_size = 1000
    overlap = 100
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
        
    # 3. Store
    ids = [f"{trip_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": "upload", "trip_id": trip_id} for _ in chunks]
    
    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )
    
    # Force persist for ChromaDB 0.3.x (DuckDB)
    try:
        chroma_client.persist()
    except AttributeError:
        pass # Chroma 0.4.x persists automatically

@app.post("/upload_trip")
async def upload_trip(
    trip_id: str = Form(...),
    file: UploadFile = File(...),
    admin_key: str = Form(...)
):
    # Simple security check
    if admin_key != os.getenv("ADMIN_PASSWORD", "secret123"):
        raise HTTPException(status_code=401, detail="Senha incorreta")
        
    try:
        # Read file into memory
        content = await file.read()
        file_stream = BytesIO(content)
        
        process_pdf_upload(file_stream, trip_id.upper())
        
        return {"status": "success", "trip_id": trip_id, "message": "Trip created and PDF indexed."}
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health_check():
    return {"status": "GoNavigator Travel Concierge is Online"}

@app.get("/webhook")
async def verify_webhook(request: Request):
    """Handles the Meta Webhook Verification Challenge."""
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode and token:
        if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
            return PlainTextResponse(content=challenge, status_code=200)
    raise HTTPException(status_code=403, detail="Verification failed")

@app.post("/webhook")
async def receive_message(request: Request):
    """Receives incoming WhatsApp messages."""
    data = await request.json()
    
    try:
        entry = data.get('entry', [])[0]
        changes = entry.get('changes', [])[0]
        value = changes.get('value', {})
        
        if 'messages' in value:
            message = value['messages'][0]
            from_number = message['from']
            msg_body = message.get('text', {}).get('body', '').strip()
            
            logger.info(f"From {from_number}: {msg_body}")
            
            # SESSION LOGIC
            sessions = load_sessions()
            user_trip_id = sessions.get(from_number)
            
            if not user_trip_id:
                # 1. New User Flow -> Ask for Trip Code
                # Check if the message LOOKS like a trip code (e.g., simplistic check)
                if len(msg_body) < 15 and msg_body.isalnum():
                     # Assume they are trying to login
                     # In a real app, verify trip_id exists in DB first
                     trip_code = msg_body.upper()
                     save_session(from_number, trip_code)
                     send_whatsapp_message(from_number, f"Bem-vindo! Código {trip_code} registrado. Pode perguntar sobre sua viagem.")
                else:
                    send_whatsapp_message(from_number, "Olá! Sou o assistente virtual da sua agência. Por favor, digite o *Código da Viagem* (ex: PARIS24) para começar.")
            else:
                # 2. Authenticated User Flow -> RAG
                handle_qa(from_number, user_trip_id, msg_body)
                
    except Exception as e:
        logger.error(f"Error processing webhook: {e}", exc_info=True)
        
    return {"status": "processed"}

def handle_qa(user_phone: str, trip_id: str, query: str):
    """Runs the RAG flow filtered by Trip ID."""
    
    # 1. Search with Metadata Filter
    results = collection.query(
        query_texts=[query],
        n_results=3,
        where={"trip_id": trip_id} # CRITICAL: This isolates the trips
    )
    
    # Check if we found anything (Reliability/Handoff Check)
    # Chroma returns distances. High distance = bad match.
    # For now, we check if documents list is empty or generic heuristic.
    if not results['documents'][0]:
        send_handoff_message(user_phone, query)
        return

    context = "\n".join(results['documents'][0])
    
    # 2. Prompt
    prompt = (
        f"Você é um guia de viagem amigável e útil para a viagem código {trip_id}. "
        f"Use SOMENTE o contexto abaixo (que é o roteiro oficial em PDF) para responder. "
        f"Se a informação não estiver explicita no contexto, DIGA: 'HANDOFF_REQUIRED'. "
        f"Responda em Português do Brasil.\n\n"
        f"Contexto do Roteiro:\n{context}\n\n"
        f"Pergunta do Viajante: {query}"
    )
    
    try:
        response = model.generate_content(prompt)
        reply_text = response.text.strip()
        
        if "HANDOFF_REQUIRED" in reply_text:
            send_handoff_message(user_phone, query)
        else:
            send_whatsapp_message(user_phone, reply_text)
            
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        send_whatsapp_message(user_phone, "Desculpe, tive um erro técnico. Tente novamente em instantes.")

def send_handoff_message(user_phone: str, query: str):
    """Handles the Handoff logic."""
    # 1. Tell User
    send_whatsapp_message(user_phone, "Não encontrei essa informação precisa no roteiro. Já notifiquei o agente responsável para te ajudar!")
    
    # 2. Log for Admin (Or email/WhatsApp to Admin)
    # We append to a local file for this MVP
    with open("concierge_log.txt", "a") as f:
        f.write(f"URGENT: User {user_phone} asked: '{query}'\n")

def send_whatsapp_message(to_number: str, text: str):
    """Sends a text message back to the user via WhatsApp Cloud API."""
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
