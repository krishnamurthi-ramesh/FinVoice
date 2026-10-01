import os
from frontend.app import ui
from dotenv import load_dotenv

load_dotenv()

# HuggingFace Spaces requires an app.py file in the root directory
if __name__ == "__main__":
    print("🚀 Starting FinVoice on HuggingFace Spaces...")
    ui.launch(server_name="0.0.0.0", server_port=7860)
