import os
import whisper
from gtts import gTTS
from tempfile import NamedTemporaryFile
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from app.agent.state import AgentState
from tools.agent_tools import get_customer_profile, get_loan_status, calculate_emi, get_kyc_status, escalate_to_human
from rag.retriever import retrieve_knowledge
from dotenv import load_dotenv

load_dotenv()

from groq import Groq

# Use Groq client for Whisper transcription directly to avoid local ffmpeg dependency
groq_api_key = os.environ.get("GROQ_API_KEY", "")
groq_client = Groq(api_key=groq_api_key)

llm = ChatGroq(model="openai/gpt-oss-120b", groq_api_key=groq_api_key)

def stt_node(state: AgentState) -> AgentState:
    """Transcribes audio using Groq's lightning-fast Whisper API."""
    try:
        with open(state["audio_path"], "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(state["audio_path"], file.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
            )
            
        transcript_text = transcription.text.strip()
        detected_lang = getattr(transcription, "language", "en")
        
        # Mock confidence check based on transcript length
        confidence = 0.9 if len(transcript_text) > 2 else 0.4
        
        return {
            "transcript": transcript_text,
            "language": detected_lang,
            "confidence": confidence
        }
    except Exception as e:
        return {"error": f"Groq STT Error: {str(e)}"}

def intent_node(state: AgentState) -> AgentState:
    """Classifies intent and checks if fallback/handoff is needed."""
    if state.get("confidence", 1.0) < 0.6:
        return {"intent": "unknown", "requires_human": True}
        
    text = state.get("transcript", "")
    prompt = PromptTemplate.from_template(
        "Classify intent into: [loan_status, check_kyc, calculate_emi, policy_question, human_agent, general]. Text: {text}\nIntent:"
    )
    
    try:
        intent = (prompt | llm).invoke({"text": text}).content.strip().lower()
        if "human" in intent or intent == "unknown":
            return {"intent": "human_agent", "requires_human": True}
        return {"intent": intent, "requires_human": False}
    except:
        return {"intent": "general"}

def tool_execution_node(state: AgentState) -> AgentState:
    """Executes relevant tools based on intent and customer context."""
    intent = state.get("intent")
    cid = state.get("customer_id", "C1024")
    
    # Auto-load customer context if empty
    ctx = state.get("customer_context", {})
    if not ctx:
        ctx = get_customer_profile(cid)
        
    action_result = ""
    
    if intent == "loan_status":
        action_result = str(get_loan_status(cid))
    elif intent == "check_kyc":
        action_result = str(get_kyc_status(cid))
    elif intent == "calculate_emi":
        # Extracting entities robustly requires LLM tool binding, keeping it simple here
        action_result = "Please specify amount and rate for EMI (Default mocked: " + str(calculate_emi(100000, 10, 12)) + ")"
    elif intent == "policy_question":
        action_result = retrieve_knowledge(state.get("transcript", ""))
    
    return {
        "customer_context": ctx,
        "last_action": f"Executed {intent}: {action_result}",
        "messages": [("system", f"Tool Result: {action_result}")]
    }

def handoff_node(state: AgentState) -> AgentState:
    """Escalates to a human agent."""
    cid = state.get("customer_id", "C1024")
    summary = state.get("transcript", "Unknown issue")
    ticket = escalate_to_human(cid, summary)
    
    return {
        "ticket_id": ticket["ticket_id"],
        "response_text": f"I have created a support ticket ({ticket['ticket_id']}) and escalated this to a human agent. They will contact you shortly."
    }

def generation_node(state: AgentState) -> AgentState:
    """Generates natural language response grounded in context and tool results."""
    if state.get("requires_human"):
        # Skip generation if we already generated the handoff text
        return {}
        
    ctx = state.get("customer_context", {})
    tool_result = state.get("last_action", "")
    transcript = state.get("transcript", "")
    lang = state.get("language", "en")
    
    prompt = f"""
    You are an enterprise AI Voice Assistant for a BFSI institution named FinVoice.
    
    CRITICAL INSTRUCTION: You MUST generate your final response entirely in the language corresponding to this ISO code: '{lang}'.
    If '{lang}' is an Indian regional language like Tamil or Hindi, DO NOT use pure, highly formal, or academic language ("Thooya Tamil" or "Shuddh Hindi"). 
    Instead, use natural, everyday colloquial spoken slang (the way normal people actually speak on the phone).
    If '{lang}' is not English, translate your response to '{lang}' before outputting.
    
    Customer Context: {ctx}
    Tool Data: {tool_result}
    User Query: {transcript}
    
    Respond helpfully and concisely to the user based ONLY on the Tool Data and Context. 
    Format for Text-To-Speech (keep it conversational, max 2 sentences).
    """
    
    try:
        response = llm.invoke(prompt)
        return {"response_text": response.content}
    except Exception as e:
        return {"error": str(e)}

import edge_tts
import asyncio

async def _generate_audio(text, voice, output_path):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def tts_node(state: AgentState) -> AgentState:
    """Converts response to speech using Microsoft Azure Neural TTS (Edge-TTS)."""
    text = state.get("response_text", "")
    lang = str(state.get("language", "en")).lower()
    
    # Map ISO codes AND full language names to ultra-realistic Azure Neural Voices
    voice_map = {
        "en": "en-IN-NeerjaNeural", "english": "en-IN-NeerjaNeural",
        "hi": "hi-IN-SwaraNeural", "hindi": "hi-IN-SwaraNeural",
        "ta": "ta-IN-PallaviNeural", "tamil": "ta-IN-PallaviNeural",
        "te": "te-IN-ShrutiNeural", "telugu": "te-IN-ShrutiNeural",
        "mr": "mr-IN-AarohiNeural", "marathi": "mr-IN-AarohiNeural",
        "ml": "ml-IN-SobhanaNeural", "malayalam": "ml-IN-SobhanaNeural",
        "gu": "gu-IN-DhwaniNeural", "gujarati": "gu-IN-DhwaniNeural",
        "bn": "bn-IN-TanishaaNeural", "bengali": "bn-IN-TanishaaNeural"
    }
    
    voice = voice_map.get(lang, "en-US-AriaNeural") # Fallback to US English if unknown
    
    try:
        out = NamedTemporaryFile(suffix=".mp3", delete=False)
        out.close() # Close so edge-tts can write to it
        
        # Safely run the async edge_tts module in our sync node
        asyncio.run(_generate_audio(text, voice, out.name))
        
        return {"response_audio_path": out.name}
    except Exception as e:
        return {"error": f"TTS Error: {str(e)}"}
