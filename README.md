# 🏦 FinVoice AI - Production BFSI Voice Agent

An autonomous, multi-lingual AI voice agent built for the Banking, Financial Services, and Insurance (BFSI) sector. This system is designed as a Forward Deployed Engineering (FDE) prototype that routes real-time voice queries using a stateful agent graph, processes Retrieval-Augmented Generation (RAG) for banking policies, calls mock REST APIs for customer data, and gracefully escalates frustrated customers to human agents.

## 🌟 Key Capabilities
* **Stateful Reasoning (LangGraph):** Uses a deterministic Directed Acyclic Graph (DAG) to guarantee reliable API routing and avoid infinite AI loops.
* **Multilingual Voice-to-Voice:** 
  * **STT**: Leverages Groq's `whisper-large-v3-turbo` for instantaneous multilingual transcription.
  * **TTS**: Uses Microsoft Azure Neural TTS (`edge-tts`) to generate hyper-realistic, native-sounding Indian regional accents (Hindi, Tamil, Marathi, Telugu, etc.)
* **RAG Vector Search:** Uses HuggingFace Embeddings (`all-MiniLM-L6-v2`) and FAISS to retrieve real banking policies locally.
* **Contextual API Tool-Calling:** Executes backend mock functions (e.g. `get_loan_status`, `check_kyc`) dynamically based on user intent.
* **Dynamic Slang Generation:** Employs a 120-Billion parameter LLM (`openai/gpt-oss-120b` via Groq) specifically prompted to generate colloquial conversational dialects rather than robotic formal text.

## 🚀 Tech Stack
* **Agent Framework:** LangGraph, LangChain
* **LLM:** `gpt-oss-120b` (Groq API)
* **Backend:** FastAPI, Python 3.10+
* **Frontend:** Gradio (Enterprise UI Theme)
* **RAG:** FAISS, HuggingFace Sentence Transformers
* **Audio:** Groq Whisper API, Edge-TTS

## ⚙️ Installation & Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/FinVoice.git
   cd FinVoice
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install edge-tts
   ```
3. Setup environment variables (`.env`):
   ```env
   GROQ_API_KEY=your_groq_key
   ```
4. Run the Web Application:
   ```bash
   python main.py --mode ui
   ```
5. Run automated Multilingual tests:
   ```bash
   python test_multilingual.py
   ```

## 🏗️ Architecture
The core agent logic runs inside `app/agent/nodes.py` which defines 5 specific nodes:
1. `stt_node`: Ingests audio, calls Whisper API, detects language.
2. `intent_node`: Uses LLM to classify if the query needs a Tool, Policy (RAG), or Human.
3. `tool_execution_node`: Routes dynamically to backend Python functions.
4. `generation_node`: Translates the result into the detected language using colloquial slang.
5. `tts_node`: Generates Microsoft Azure Neural Voice files.
