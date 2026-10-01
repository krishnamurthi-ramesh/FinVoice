import gradio as gr
from app.agent.graph import bfsi_agent

def process_voice_query(audio_path, customer_id):
    if not audio_path:
        return "No audio provided.", None, ""
        
    initial_state = {
        "audio_path": audio_path, 
        "customer_id": customer_id,
        "messages": []
    }
    
    try:
        # Run the full agent pipeline
        result = bfsi_agent.invoke(initial_state)
        
        if result.get("error"):
            return f"❌ Error: {result['error']}", None, ""
            
        transcript = result.get("transcript", "N/A")
        intent = result.get("intent", "unknown")
        response_text = result.get("response_text", "N/A")
        audio_out = result.get("response_audio_path", None)
        ticket = result.get("ticket_id", "None")
        
        # Format the debug/observability log for the UI
        debug_log = f"🎤 **Transcript**: {transcript}\n"
        debug_log += f"🧠 **Detected Intent**: `{intent.upper()}`\n"
        debug_log += f"🧑‍💼 **Customer Profile Used**: {customer_id}\n"
        if ticket != "None":
            debug_log += f"🎫 **Escalation Ticket Generated**: {ticket}\n"
            
        return response_text, audio_out, debug_log
        
    except Exception as e:
        return f"Pipeline Error: {str(e)}", None, "Traceback failed."

# Apply a polished, enterprise-style theme suitable for BFSI
bfsi_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate"
)

with gr.Blocks(theme=bfsi_theme, title="FinVoice: Production BFSI Support") as ui:
    gr.Markdown(
        """
        # 🏦 FinVoice: Production BFSI Support
        *Powered by LangGraph, Groq, FAISS, and Whisper*
        
        Demonstrating context-aware voice AI that executes real workflows instead of just chatting.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            customer_selector = gr.Dropdown(
                choices=["C1024", "UNKNOWN"], 
                value="C1024", 
                label="Simulated Caller ID (Context Injection)"
            )
            audio_input = gr.Audio(type="filepath", label="🗣️ Speak to the Agent (Mock Mic/Upload)")
            submit_btn = gr.Button("🎙️ Process Query", variant="primary")
            
        with gr.Column(scale=2):
            agent_text_output = gr.Textbox(label="Agent Text Response", interactive=False, lines=4)
            agent_voice_output = gr.Audio(label="Agent Voice Response", interactive=False)
            
    with gr.Accordion("⚙️ Backend Trace / Pipeline Observability", open=True):
        debug_output = gr.Markdown("Waiting for input...")

    submit_btn.click(
        fn=process_voice_query,
        inputs=[audio_input, customer_selector],
        outputs=[agent_text_output, agent_voice_output, debug_output]
    )
