from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import shutil
import os
from tempfile import NamedTemporaryFile
from pydantic import BaseModel

from backend.services.mock_db import MOCK_CUSTOMERS, MOCK_LOANS, MOCK_KYC, MOCK_TICKETS
from tools.agent_tools import create_support_ticket
from app.agent.graph import bfsi_agent

app = FastAPI(
    title="BFSI Voice Agent API", 
    description="Production-Oriented Customer Support & Onboarding API",
    version="2.0.0"
)

# --- Voice Agent Endpoint ---

@app.post("/api/v1/voice/chat")
async def voice_chat_endpoint(audio: UploadFile = File(...), customer_id: str = Form("C1024")):
    """End-to-end voice AI agent pipeline with context awareness."""
    temp_audio = NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        shutil.copyfileobj(audio.file, temp_audio)
        temp_audio.close()
        
        initial_state = {
            "audio_path": temp_audio.name, 
            "customer_id": customer_id,
            "messages": []
        }
        
        result = bfsi_agent.invoke(initial_state)
        
        if result.get("error"):
            return JSONResponse(status_code=500, content={"error": result["error"]})
            
        return FileResponse(
            result.get("response_audio_path", ""), 
            media_type="audio/mpeg", 
            headers={
                "X-Transcript": str(result.get("transcript", "")),
                "X-Intent": str(result.get("intent", "")),
                "X-Language": str(result.get("language", "en")),
                "X-Response-Text": str(result.get("response_text", "")).replace("\n", " "),
                "X-Ticket-ID": str(result.get("ticket_id", "")),
                "X-Requires-Human": str(result.get("requires_human", False))
            }
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(temp_audio.name):
            os.remove(temp_audio.name)

# --- Mock BFSI API Endpoints ---

@app.get("/customer/{customer_id}")
def get_customer(customer_id: str):
    if customer_id not in MOCK_CUSTOMERS:
        raise HTTPException(status_code=404, detail="Customer not found")
    return MOCK_CUSTOMERS[customer_id]

@app.get("/loan/{loan_id}")
def get_loan(loan_id: str):
    if loan_id not in MOCK_LOANS:
        raise HTTPException(status_code=404, detail="Loan not found")
    return MOCK_LOANS[loan_id]

@app.get("/loan/{loan_id}/status")
def get_loan_status(loan_id: str):
    loan = get_loan(loan_id)
    return {"loan_id": loan_id, "status": loan["status"]}

@app.get("/loan/{loan_id}/emi")
def get_loan_emi(loan_id: str):
    loan = get_loan(loan_id)
    return {"loan_id": loan_id, "emi": loan["emi"], "next_emi_date": loan["next_emi_date"]}

@app.get("/kyc/{customer_id}/status")
def get_kyc_status_api(customer_id: str):
    if customer_id not in MOCK_KYC:
        raise HTTPException(status_code=404, detail="KYC not found")
    return MOCK_KYC[customer_id]

class SupportTicket(BaseModel):
    customer_id: str
    reason: str
    priority: str = "normal"

@app.post("/support/ticket")
def create_ticket(ticket: SupportTicket):
    return create_support_ticket(ticket.customer_id, ticket.reason, ticket.priority)

@app.post("/handoff")
def trigger_handoff(customer_id: str, summary: str):
    ticket = create_support_ticket(customer_id, summary, priority="high")
    return {"message": "Escalated to human agent", "ticket": ticket}
