import json
import numpy as np
from typing import List, Dict
import requests
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

class RAGSystem:
    def __init__(self, groq_api_key=None):
        """
        Initialize RAG system with API keys
        
        Args:
            groq_api_key: Groq API key for LLM (get free at console.groq.com)
        """
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        self.chunks = []
        self.embeddings = []
        
        # Load local sentence transformer model (no API needed!)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("✓ Embedding model loaded")
        
        # Groq LLM endpoint (free, very fast)
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        
    def load_chunks(self, filename='processed_chunks.json'):
        """Load processed chunks from JSON"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            print(f"✓ Loaded {len(self.chunks)} chunks from {filename}")
            return True
        except FileNotFoundError:
            print(f"✗ File {filename} not found!")
            return False
    
    def get_embedding(self, text):
        """Get embedding vector for text using local SentenceTransformer model"""
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return np.array(embedding)
        except Exception as e:
            print(f"Error getting embedding: {str(e)}")
            return None
    
    def create_embeddings(self):
        """Create embeddings for all chunks"""
        print("\n" + "="*50)
        print("CREATING EMBEDDINGS")
        print("="*50)
        
        self.embeddings = []
        
        for i, chunk in enumerate(self.chunks, 1):
            print(f"Embedding chunk {i}/{len(self.chunks)}", end='\r')
            
            embedding = self.get_embedding(chunk['text'])
            if embedding is not None:
                self.embeddings.append(embedding)
            else:
                print(f"\n✗ Failed to embed chunk {i}")
        
        print(f"\n✓ Created {len(self.embeddings)} embeddings")
        
    def save_embeddings(self, filename='embeddings.json'):
        """Save embeddings to file"""
        embeddings_data = {
            'embeddings': [emb.tolist() for emb in self.embeddings],
            'chunks': self.chunks
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(embeddings_data, f)
        print(f"✓ Embeddings saved to {filename}")
    
    def load_embeddings(self, filename='embeddings.json'):
        """Load pre-computed embeddings"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.embeddings = [np.array(emb) for emb in data['embeddings']]
            self.chunks = data['chunks']
            print(f"✓ Loaded {len(self.embeddings)} embeddings from {filename}")
            return True
        except FileNotFoundError:
            print(f"✗ File {filename} not found!")
            return False
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2)
    
    def retrieve_relevant_chunks(self, query, top_k=3):
        """
        Retrieve most relevant chunks for a query
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve
        """
        print(f"\nRetrieving relevant chunks for: '{query}'")
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        if query_embedding is None:
            return []
        
        # Calculate similarities
        similarities = []
        for i, chunk_embedding in enumerate(self.embeddings):
            similarity = self.cosine_similarity(query_embedding, chunk_embedding)
            similarities.append((i, similarity))
        
        # Sort by similarity and get top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in similarities[:top_k]]
        
        relevant_chunks = [self.chunks[idx] for idx in top_indices]
        
        print(f"✓ Retrieved {len(relevant_chunks)} relevant chunks")
        return relevant_chunks
    
    def generate_response(self, query, relevant_chunks):
        """
        Generate response using Groq LLM
        
        Args:
            query: User's question
            relevant_chunks: Retrieved relevant context
        """
        # Build context from relevant chunks
        context = "\n\n".join([
            f"Source: {chunk['source_title']}\n{chunk['text']}"
            for chunk in relevant_chunks
        ])
        
        # Create prompt
        prompt = f"""You are a helpful assistant for the University of Mines and Technology (UMat) website. 
Answer the user's question based ONLY on the following context from the UMat website.
If the answer is not in the context, say "I don't have that information in the available content."

Context:
{context}

Question: {query}

Answer:"""
        
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.3-70b-versatile",  # Fast and free on Groq
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(
                self.groq_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            answer = result['choices'][0]['message']['content']
            
            return {
                'answer': answer,
                'sources': [chunk['source_url'] for chunk in relevant_chunks]
            }
            
        except Exception as e:
            return {
                'answer': f"Error generating response: {str(e)}",
                'sources': []
            }
    
    def query(self, question, top_k=3):
        """
        Main query function - retrieve and generate answer
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
        """
        print("\n" + "="*50)
        print("PROCESSING QUERY")
        print("="*50)
        
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(question, top_k)
        
        if not relevant_chunks:
            return {
                'answer': "Sorry, I couldn't retrieve relevant information.",
                'sources': []
            }
        
        # Generate response
        print("Generating response...")
        response = self.generate_response(question, relevant_chunks)
        
        return response


# Usage Example
if __name__ == "__main__":
    # Initialize RAG system
    rag = RAGSystem(
        groq_api_key=os.getenv('GROQ_API_KEY')  # Get from console.groq.com
    )
    
    # Load chunks
    rag.load_chunks('processed_chunks.json')
    
    # Create embeddings (do this once, then save)
    rag.create_embeddings()
    rag.save_embeddings('embeddings.json')
    
    # Or load pre-computed embeddings
    # rag.load_embeddings('embeddings.json')
    
    # Test queries
    test_questions = [
        "How do I apply for admission to UMat?",
        "What programs does UMat offer?",
        "Where is UMat located?",
        "What are the admission requirements?"
    ]
    
    for question in test_questions:
        result = rag.query(question, top_k=3)
        print(f"\nQ: {question}")
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print("-" * 50)