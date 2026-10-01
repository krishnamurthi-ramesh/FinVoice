from typing import TypedDict, Annotated, Optional
import operator

class AgentState(TypedDict):
    # Inputs & Speech
    audio_path: str
    transcript: str
    language: str
    confidence: float
    
    # Customer Context
    customer_id: str
    customer_context: dict
    
    # Conversation State
    intent: str
    messages: Annotated[list, operator.add]
    
    # Agent Workflow
    last_action: str
    requires_human: bool
    ticket_id: Optional[str]
    
    # Outputs
    response_text: str
    response_audio_path: str
    error: str
