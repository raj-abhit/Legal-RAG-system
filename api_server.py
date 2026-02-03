"""
FastAPI Backend for Indian Law Chatbot
Connects frontend.html with the Legal RAG System
Supports real-time streaming responses
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, AsyncGenerator
import asyncio
import json
import os
from dotenv import load_dotenv

# Import your Legal RAG System
from legal_rag_system import LegalRAGSystem

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Indian Law Chatbot API",
    description="RAG-powered legal assistant for Indian law",
    version="2.1.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system (load once at startup)
rag_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system when server starts"""
    global rag_system
    print("🚀 Starting Indian Law Chatbot API...")
    print("📚 Loading Legal RAG System...")
    
    try:
        rag_system = LegalRAGSystem()
        
        # Check if vectorstore exists
        if os.path.exists("legal_vectorstore"):
            rag_system.load_vectorstore()
            print("✓ Vectorstore loaded successfully!")
        else:
            print("⚠ No vectorstore found. Please run setup first.")
            print("Run: python interactive_legal_rag_v2.py")
            
        print("✓ Legal RAG System ready!")
        print("🌐 API server running at http://localhost:8000")
        print("📄 Frontend available at http://localhost:8000/")
        
    except Exception as e:
        print(f"✗ Error initializing RAG system: {str(e)}")
        print("Please ensure vectorstore exists and .env file has GROQ_API_KEY")

# Request/Response models
class QueryRequest(BaseModel):
    question: str
    research_mode: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    question: str

class HealthResponse(BaseModel):
    status: str
    rag_loaded: bool
    message: str

# Routes
@app.get("/", response_class=FileResponse)
async def serve_frontend():
    """Serve the frontend HTML"""
    return FileResponse("frontend.html")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rag_loaded": rag_system is not None and rag_system.vectorstore is not None,
        "message": "Indian Law Chatbot API is running"
    }

@app.post("/api/query", response_model=QueryResponse)
async def query_legal_system(request: QueryRequest):
    """
    Query the legal RAG system
    
    Args:
        question: The legal question to ask
        research_mode: Whether to use RAG (True) or just LLM (False)
    
    Returns:
        QueryResponse with answer and sources
    """
    if not rag_system or not rag_system.vectorstore:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized. Please load documents first."
        )
    
    try:
        # Query the RAG system
        result = rag_system.query(request.question)
        
        return QueryResponse(
            answer=result['answer'],
            sources=result.get('sources', []),
            question=request.question
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/api/query/stream")
async def stream_query(question: str):
    """
    Stream the response for real-time updates
    This creates a better user experience with progressive loading
    """
    if not rag_system or not rag_system.vectorstore:
        return StreamingResponse(
            iter(["Error: RAG system not initialized"]),
            media_type="text/plain"
        )
    
    async def generate_response() -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        try:
            # Get the full response
            result = rag_system.query(question)
            answer = result['answer']
            sources = result.get('sources', [])
            
            # Stream the answer word by word for effect
            words = answer.split()
            for i, word in enumerate(words):
                yield f"{word} "
                await asyncio.sleep(0.05)  # Small delay for streaming effect
            
            # Add sources at the end
            if sources:
                yield "\n\n📚 Sources:\n"
                for i, source in enumerate(sources[:3], 1):
                    yield f"{i}. {source}\n"
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            yield f"Error: {str(e)}"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/plain"
    )

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat
    Better for interactive conversations
    """
    await websocket.accept()
    
    if not rag_system or not rag_system.vectorstore:
        await websocket.send_json({
            "type": "error",
            "message": "RAG system not initialized"
        })
        await websocket.close()
        return
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            question = data.get("question", "")
            
            if not question:
                await websocket.send_json({
                    "type": "error",
                    "message": "No question provided"
                })
                continue
            
            # Send "thinking" status
            await websocket.send_json({
                "type": "status",
                "message": "Searching legal documents..."
            })
            
            try:
                # Query the RAG system
                result = rag_system.query(question)
                
                # Send the complete response
                await websocket.send_json({
                    "type": "answer",
                    "answer": result['answer'],
                    "sources": result.get('sources', []),
                    "question": question
                })
                
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error processing query: {str(e)}"
                })
                
    except WebSocketDisconnect:
        print("Client disconnected from WebSocket")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        await websocket.close()

@app.get("/api/suggestions")
async def get_suggestions():
    """Get suggested queries for the frontend"""
    return {
        "suggestions": [
            {
                "icon": "summarize",
                "text": "Kesavananda Bharati case",
                "query": "What is the Kesavananda Bharati case and its significance?"
            },
            {
                "icon": "gavel",
                "text": "Bail provisions under BNSS",
                "query": "What are the bail provisions under BNSS 2023?"
            },
            {
                "icon": "security",
                "text": "Article 21 privacy details",
                "query": "Explain Article 21 and the right to privacy in India"
            },
            {
                "icon": "update",
                "text": "Recent amendments",
                "query": "What are the recent changes in Indian criminal law (BNS, BNSS, BSA)?"
            }
        ]
    }

# Run with: uvicorn api_server:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )








