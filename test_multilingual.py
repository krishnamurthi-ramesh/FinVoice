import os
from gtts import gTTS
from app.agent.graph import bfsi_agent
from dotenv import load_dotenv

load_dotenv()

def create_mock_customer_audio(text, lang, filename):
    print(f"Simulating customer speaking {lang}...")
    tts = gTTS(text=text, lang=lang)
    tts.save(filename)
    return filename

def run_test(audio_path, test_name):
    print(f"\nRunning BFSI Agent Pipeline: {test_name}")
    initial_state = {
        "audio_path": audio_path, 
        "customer_id": "C1024",
        "messages": []
    }
    
    result = bfsi_agent.invoke(initial_state)
    
    # Safely print without emojis to avoid Windows cp1252 codec errors
    print(f"STT Transcript: {result.get('transcript')}")
    print(f"Language Detected: {result.get('language')}")
    print(f"Intent Detected: {result.get('intent')}")
    print(f"Agent Response: {result.get('response_text')}")
    print(f"Agent Audio File Generated: {result.get('response_audio_path')}")
    print("-" * 50)

if __name__ == "__main__":
    print("Starting Multilingual Agent Evaluation...\n")
    
    # 1. Hindi Test (Tool Calling Intent)
    hi_audio = create_mock_customer_audio("मेरे लोन का स्टेटस क्या है?", "hi", "mock_hindi.mp3")
    run_test(hi_audio, "Hindi Loan Status Query")
    
    # 2. Tamil Test (RAG Knowledge Intent)
    ta_audio = create_mock_customer_audio("தனிநபர் கடனுக்கான வட்டி விகிதம் என்ன?", "ta", "mock_tamil.mp3")
    run_test(ta_audio, "Tamil Interest Rate Query")
    
    print("Multilingual tests completed successfully!")
