import gradio as gr
import whisper  # Corrected import for Whisper
from gtts import gTTS
import os
from groq import Groq
from tempfile import NamedTemporaryFile

# Initialize Groq API Client
client = Groq(
    api_key="your_api_key",
)

import whisper

# Load the Whisper model
model = whisper.load_model("base")  

def speech_to_text(audio_path):
    transcription = model.transcribe(audio_path)["text"]
    return transcription

def generate_response(text):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": text,
            }
        ],
     model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content

def text_to_speech(text):
    tts = gTTS(text)
    output_audio = NamedTemporaryFile(suffix=".mp3", delete=False)
    tts.save(output_audio.name)
    return output_audio.name

def chatbot_pipeline(audio_path):
    try:
       #Convert speech to text
        text_input = speech_to_text(audio_path)

        #response from LLaMA model
        response_text = generate_response(text_input)

        #Convert response text to speech
        response_audio_path = text_to_speech(response_text)

        return response_text, response_audio_path

    except Exception as e:
        return str(e), None

# Create Gradio Interface
iface = gr.Interface(
    fn=chatbot_pipeline,
    inputs=gr.Audio(type="filepath", label="Speak"), 
    outputs=[
        gr.Textbox(label="Response Text"),
        gr.Audio(label="Response Audio")
    ],
    title="Real-Time Voice-to-Voice Chatbot"
)

iface.launch()
