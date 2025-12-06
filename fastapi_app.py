from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import numpy as np
import requests
import os
from typing import List, Optional
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="UMat RAG Chatbot API",
    description="RAG-based chatbot for University of Mines and Technology website",
    version="1.0.0",
    docs_url='/docs'
)

# Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for RAG system
chunks = []
embeddings = []
embedding_model = None
groq_api_key = os.getenv('GROQ_API_KEY')

# Groq LLM endpoint (free, very fast)
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    relevant_chunks: List[dict]

class HealthResponse(BaseModel):
    status: str
    chunks_loaded: int
    embeddings_loaded: int
    api_keys_configured: dict


# Helper functions
def load_data():
    """Load embeddings and chunks on startup"""
    global chunks, embeddings
    
    try:
        with open('embeddings.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        embeddings = [np.array(emb) for emb in data['embeddings']]
        chunks = data['chunks']
        print(f"✓ Loaded {len(chunks)} chunks and {len(embeddings)} embeddings")
        return True
    except FileNotFoundError:
        print("✗ embeddings.json not found! Run the RAG system first to create embeddings.")
        return False

def get_embedding(text):
    """Get embedding vector for text using local SentenceTransformer model"""
    global embedding_model
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")
    
    try:
        embedding = embedding_model.encode(text, convert_to_numpy=True)
        return np.array(embedding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def retrieve_relevant_chunks(query, top_k=3):
    """Retrieve most relevant chunks for a query"""
    query_embedding = get_embedding(query)
    
    similarities = []
    for i, chunk_embedding in enumerate(embeddings):
        similarity = cosine_similarity(query_embedding, chunk_embedding)
        similarities.append((i, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in similarities[:top_k]]
    
    return [chunks[idx] for idx in top_indices]

def generate_response(query, relevant_chunks):
    """Generate response using Groq LLM"""
    context = "\n\n".join([
        f"Source: {chunk['source_title']}\n{chunk['text']}"
        for chunk in relevant_chunks
    ])
    
    prompt = f"""You are a helpful assistant for the University of Mines and Technology (UMat) website. 
Answer the user's question based ONLY on the following context from the UMat website.
If the answer is not in the context, say "I don't have that information in the available content."
Be concise and helpful in your response.

Context:
{context}

Question: {query}

Answer:"""
    
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Load data and initialize embedding model when the API starts"""
    global embedding_model
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("✓ Embedding model loaded")
    load_data()

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to UMat RAG Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "POST /query": "Ask a question about UMat",
            "GET /health": "Check system health",
            "GET /stats": "Get system statistics",
            "docs": "/docs",
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check system health and configuration"""
    return {
        "status": "healthy" if chunks and embeddings else "unhealthy",
        "chunks_loaded": len(chunks),
        "embeddings_loaded": len(embeddings),
        "api_keys_configured": {
            "groq": bool(groq_api_key),
            "embedding_model": embedding_model is not None
        }
    }

@app.get("/stats", tags=["System"])
async def get_stats():
    """Get system statistics"""
    if not chunks:
        raise HTTPException(status_code=503, detail="No data loaded")
    
    unique_sources = len(set(chunk['source_url'] for chunk in chunks))
    avg_chunk_length = sum(len(chunk['text']) for chunk in chunks) / len(chunks)
    
    return {
        "total_chunks": len(chunks),
        "total_embeddings": len(embeddings),
        "unique_sources": unique_sources,
        "average_chunk_length": round(avg_chunk_length, 2),
        "embedding_dimension": len(embeddings[0]) if embeddings else 0
    }

@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_chatbot(request: QueryRequest):
    """
    Ask a question about UMat
    
    - **question**: Your question about UMat
    - **top_k**: Number of relevant chunks to retrieve (default: 3)
    """
    if not chunks or not embeddings:
        raise HTTPException(status_code=503, detail="System not ready. Embeddings not loaded.")
    
    if not groq_api_key:
        raise HTTPException(status_code=500, detail="Groq API key not configured")
    
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(request.question, request.top_k)
        
        if not relevant_chunks:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": [],
                "relevant_chunks": []
            }
        
        # Generate response
        answer = generate_response(request.question, relevant_chunks)
        
        return {
            "answer": answer,
            "sources": list(set(chunk['source_url'] for chunk in relevant_chunks)),
            "relevant_chunks": [
                {
                    "text": chunk['text'][:200] + "...",
                    "source_title": chunk['source_title'],
                    "source_url": chunk['source_url']
                }
                for chunk in relevant_chunks
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)