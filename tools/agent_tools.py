from backend.services.mock_db import MOCK_CUSTOMERS, MOCK_LOANS, MOCK_KYC, MOCK_TICKETS
import uuid

def get_customer_profile(customer_id: str) -> dict:
    """Retrieves customer profile information."""
    return MOCK_CUSTOMERS.get(customer_id, {"error": "Customer not found"})

def get_loan_status(customer_id: str) -> dict:
    """Retrieves active loans and their status for a given customer."""
    customer_loans = [loan for loan in MOCK_LOANS.values() if loan["customer_id"] == customer_id]
    if not customer_loans:
        return {"message": "No active loans found."}
    return {"loans": customer_loans}

def calculate_emi(principal: float, rate: float, tenure_months: int) -> dict:
    """Calculates the EMI for a given principal, annual interest rate, and tenure."""
    if rate == 0:
        return {"emi": principal / tenure_months}
    monthly_rate = (rate / 100) / 12
    emi = principal * monthly_rate * ((1 + monthly_rate) ** tenure_months) / (((1 + monthly_rate) ** tenure_months) - 1)
    return {"emi": round(emi, 2)}

def get_kyc_status(customer_id: str) -> dict:
    """Retrieves KYC verification status."""
    return MOCK_KYC.get(customer_id, {"error": "KYC record not found"})

def create_support_ticket(customer_id: str, issue: str, priority: str = "normal") -> dict:
    """Creates a support ticket for human escalation."""
    ticket_id = f"TKT-{uuid.uuid4().hex[:6].upper()}"
    ticket = {
        "ticket_id": ticket_id,
        "customer_id": customer_id,
        "issue": issue,
        "priority": priority,
        "status": "open"
    }
    MOCK_TICKETS.append(ticket)
    return ticket

def escalate_to_human(customer_id: str, summary: str) -> dict:
    """Escalates the current interaction to a human agent."""
    return create_support_ticket(customer_id, issue=summary, priority="high")
