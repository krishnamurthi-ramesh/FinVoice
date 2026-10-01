import argparse
import uvicorn
import os
from dotenv import load_dotenv

# Load Environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Run BFSI Voice Agent")
    parser.add_argument("--mode", choices=["ui", "api"], default="ui", 
                        help="Run 'ui' for the Web Application or 'api' for the FastAPI backend")
    parser.add_argument("--port", type=int, default=8000, 
                        help="Port to run on")
    
    args = parser.parse_args()
    
    if not os.getenv("GROQ_API_KEY"):
        print("⚠️ WARNING: GROQ_API_KEY is not set in .env! LLM features will fail.")
        
    if args.mode == "ui":
        print(f"🚀 Starting FinVoice: Production BFSI Support Web Application on port {args.port}...")
        from frontend.app import ui
        
        # HuggingFace Spaces injects SPACE_ID. If present, we must bind to 0.0.0.0 on port 7860.
        if os.getenv("SPACE_ID"):
            ui.launch(server_name="0.0.0.0", server_port=7860)
        else:
            ui.launch(server_port=args.port)
    else:
        print(f"🚀 Starting BFSI Voice Agent Production API on http://127.0.0.1:{args.port}/docs ...")
        from backend.main import app as fastapi_app
        uvicorn.run(fastapi_app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
