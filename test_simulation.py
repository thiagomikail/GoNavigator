import requests
import json

# Configuration
URL = "http://127.0.0.1:8000/webhook"
PHONE_NUMBER = "5511999998888" # Fake user number
TRIP_CODE = "PARIS24"          # The code we tested with

def send_message(text):
    print(f"\n--- User says: '{text}' ---")
    payload = {
        "object": "whatsapp_business_account",
        "entry": [{
            "id": "123456789",
            "changes": [{
                "value": {
                    "messaging_product": "whatsapp",
                    "metadata": {"display_phone_number": "123", "phone_number_id": "123"},
                    "messages": [{
                        "from": PHONE_NUMBER,
                        "id": "wamid.HBgLM...",
                        "timestamp": "167...",
                        "text": {"body": text},
                        "type": "text"
                    }]
                },
                "field": "messages"
            }]
        }]
    }
    
    try:
        # Note: In a real scenario, the server tries to call WhatsApp API to reply.
        # Since we don't have a valid token/phone_id, the server log will show an error 
        # *sending the reply*, but it should *generate* the reply in the logs first.
        r = requests.post(URL, json=payload)
        print(f"Server received message. Status: {r.status_code}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Simulating conversation...")
    # 1. Login
    send_message(TRIP_CODE) 
    
    # 2. Ask Question
    import time
    time.sleep(2)
    send_message("Qual é o roteiro do primeiro dia?")
