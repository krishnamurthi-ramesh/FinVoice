import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

class BFSIKnowledgeBase:
    def __init__(self, kb_dir="rag/knowledge_base"):
        self.kb_dir = kb_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None
        self.setup()

    def setup(self):
        # We manually read markdown files instead of DirectoryLoader for simplicity in paths
        docs = []
        if os.path.exists(self.kb_dir):
            for filename in os.listdir(self.kb_dir):
                if filename.endswith(".md"):
                    path = os.path.join(self.kb_dir, filename)
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                        # simple document structure
                        docs.append(text)
        
        if not docs:
            # Fallback if empty
            docs = ["Default BFSI knowledge base content."]
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.create_documents(docs)
        
        # Build FAISS index
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)

    def retrieve(self, query: str, k: int = 3):
        if not self.vectorstore:
            return []
        
        # We can implement a similarity threshold check here by using similarity_search_with_score
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        # FAISS L2 distance: lower score is better (more similar)
        # Assuming typical embedding distances, we can filter out poor matches.
        # For simplicity, returning top-k that meet a heuristic threshold.
        filtered_results = []
        for doc, score in results:
            if score < 0.7:  # Heuristic threshold for L2 distance (adjust based on embedding model)
                filtered_results.append(doc.page_content)
                
        return filtered_results

# Singleton instance
kb = BFSIKnowledgeBase()

def retrieve_knowledge(query: str) -> str:
    """Retrieves relevant policy information from the knowledge base."""
    results = kb.retrieve(query)
    if not results:
        return "No relevant information found in the knowledge base."
    return "\n\n".join(results)
