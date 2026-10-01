import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

def test_setup():
    print("Testing dependencies and API keys...")
    
    # 1. Load env
    load_dotenv()
    groq_key = os.getenv("GROQ_API_KEY")
    
    if not groq_key or groq_key == "your_groq_api_key_here":
        print(" Error: GROQ_API_KEY is not set in your environment or .env file.")
        print("Please create a .env file and add your Groq API key.")
        return
    else:
        print("Found GROQ_API_KEY.")
        
    # 2. Test LLM
    try:
        print("Testing ChatGroq connection...")
        llm = ChatGroq(model="openai/gpt-oss-120b", groq_api_key=groq_key)
        response = llm.invoke("Hello, say 'Groq is working!' if you can read this.")
        print(f"PASS LLM Response: {response.content}")
    except Exception as e:
        print(f"FAIL LLM Connection Failed: {e}")
        return

    # 3. Test Embeddings
    try:
        print("Testing HuggingFace Embeddings (all-MiniLM-L6-v2)...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        result = embeddings.embed_query("Hello world")
        print(f"PASS Embeddings generated successfully! Dimension: {len(result)}")
    except Exception as e:
        print(f"FAIL Embeddings Generation Failed: {e}")
        return
        
    print("\nEverything is working correctly! You can now run the evaluation or start the API.")

if __name__ == "__main__":
    test_setup()
