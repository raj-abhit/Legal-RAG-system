"""
Enhanced API Server with Session Management
Supports conversation memory and improved error handling
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, validator
from typing import Optional, Dict
from datetime import datetime, timedelta
import asyncio
import uuid
import os
from dotenv import load_dotenv

# Import enhanced RAG system
from legal_rag_system import LegalRAGSystem

load_dotenv()

app = FastAPI(
    title="Indian Law Chatbot API",
    description="Enhanced RAG-powered legal assistant with conversation memory",
    version="2.5.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session management
sessions: Dict[str, dict] = {}
SESSION_TIMEOUT = timedelta(hours=2)

# Initialize RAG system
rag_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system"""
    global rag_system
    print("=" * 70)
    print("🚀 Starting Enhanced Indian Law Chatbot API")
    print("=" * 70)
    
    try:
        rag_system = LegalRAGSystem()
        
        if os.path.exists("legal_vectorstore"):
            rag_system.load_vectorstore()
            print("✓ RAG System loaded with conversation memory!")
        else:
            print("⚠ No vectorstore found")
            print("Run: python interactive_legal_rag_v2.py to create it")
        
        print("\n" + "=" * 70)
        print("✓ API Server Ready!")
        print("=" * 70)
        print("🌐 Frontend: http://localhost:8000/")
        print("📖 API Docs: http://localhost:8000/docs")
        print("💚 Health: http://localhost:8000/health")
        print("=" * 70 + "\n")
        
        # Start cleanup task
        asyncio.create_task(cleanup_old_sessions())
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        print("Check .env file and vectorstore\n")


async def cleanup_old_sessions():
    """Cleanup expired sessions every 30 minutes"""
    while True:
        await asyncio.sleep(1800)  # 30 minutes
        now = datetime.now()
        expired = [sid for sid, data in sessions.items() 
                  if now - data['last_activity'] > SESSION_TIMEOUT]
        for sid in expired:
            del sessions[sid]
        if expired:
            print(f"Cleaned up {len(expired)} expired sessions")


def get_or_create_session(session_id: Optional[str] = None) -> str:
    """Get existing session or create new one"""
    if session_id and session_id in sessions:
        # Update last activity
        sessions[session_id]['last_activity'] = datetime.now()
        return session_id
    
    # Create new session
    new_session_id = str(uuid.uuid4())
    sessions[new_session_id] = {
        'created_at': datetime.now(),
        'last_activity': datetime.now(),
        'message_count': 0
    }
    return new_session_id


# Request/Response Models
class QueryRequest(BaseModel):
    question: str
    research_mode: bool = True
    session_id: Optional[str] = None
    
    @validator('question')
    def validate_question(cls, v):
        if not v or not v.strip():
            raise ValueError('Question cannot be empty')
        if len(v) > 1000:
            raise ValueError('Question too long (max 1000 characters)')
        return v.strip()


class QueryResponse(BaseModel):
    answer: str
    sources: list
    question: str
    session_id: str
    has_follow_up_context: bool = False


class SessionResponse(BaseModel):
    session_id: str
    message: str


class HealthResponse(BaseModel):
    status: str
    rag_loaded: bool
    active_sessions: int
    vectorstore_exists: bool
    message: str


# Routes
@app.get("/", response_class=FileResponse)
async def serve_frontend():
    """Serve frontend"""
    if os.path.exists("frontend.html"):
        return FileResponse("frontend.html")
    return {"error": "frontend.html not found"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check with session info"""
    vectorstore_exists = os.path.exists("legal_vectorstore")
    rag_loaded = rag_system is not None and rag_system.vectorstore is not None
    
    return {
        "status": "healthy" if rag_loaded else "degraded",
        "rag_loaded": rag_loaded,
        "active_sessions": len(sessions),
        "vectorstore_exists": vectorstore_exists,
        "message": "System ready!" if rag_loaded else "RAG system not loaded"
    }


@app.post("/api/session/start", response_model=SessionResponse)
async def start_session():
    """Start a new conversation session"""
    session_id = get_or_create_session()
    return {
        "session_id": session_id,
        "message": "Session created successfully"
    }


@app.post("/api/session/clear")
async def clear_session(session_id: str):
    """Clear conversation history for a session"""
    if not rag_system:
        raise HTTPException(503, "RAG system not initialized")
    
    if session_id in sessions:
        # Clear RAG system memory
        # Note: This clears global memory. For true multi-user,
        # you'd need separate RAG instances per session
        rag_system.clear_conversation()
        sessions[session_id]['last_activity'] = datetime.now()
        return {"message": "Conversation cleared", "session_id": session_id}
    
    raise HTTPException(404, "Session not found")


@app.post("/api/query", response_model=QueryResponse)
async def query_legal_system(
    request: QueryRequest,
    x_session_id: Optional[str] = Header(None)
):
    """Query with conversation memory"""
    if not rag_system or not rag_system.vectorstore:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized. Check /health for status."
        )
    
    try:
        # Get or create session
        session_id = get_or_create_session(request.session_id or x_session_id)
        
        # Update session
        sessions[session_id]['message_count'] += 1
        sessions[session_id]['last_activity'] = datetime.now()
        
        # Query with memory
        result = rag_system.query(request.question)
        
        # Check if error occurred
        if result.get('error'):
            raise HTTPException(400, result['answer'])
        
        # Check if it was a follow-up
        is_follow_up = rag_system.memory.is_follow_up(request.question)
        
        return QueryResponse(
            answer=result['answer'],
            sources=result.get('sources', []),
            question=request.question,
            session_id=session_id,
            has_follow_up_context=is_follow_up
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint with session support"""
    await websocket.accept()
    
    if not rag_system or not rag_system.vectorstore:
        await websocket.send_json({
            "type": "error",
            "message": "RAG system not initialized"
        })
        await websocket.close()
        return
    
    # Create session for this WebSocket
    session_id = get_or_create_session()
    
    try:
        await websocket.send_json({
            "type": "session",
            "session_id": session_id,
            "message": "Connected"
        })
        
        while True:
            data = await websocket.receive_json()
            question = data.get("question", "")
            
            if not question:
                await websocket.send_json({
                    "type": "error",
                    "message": "No question provided"
                })
                continue
            
            # Send thinking status
            await websocket.send_json({
                "type": "status",
                "message": "Searching documents..."
            })
            
            try:
                # Query with memory
                result = rag_system.query(question)
                
                if result.get('error'):
                    await websocket.send_json({
                        "type": "error",
                        "message": result['answer']
                    })
                else:
                    await websocket.send_json({
                        "type": "answer",
                        "answer": result['answer'],
                        "sources": result.get('sources', []),
                        "question": question,
                        "session_id": session_id
                    })
                
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error: {str(e)}"
                })
                
    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        await websocket.close()


@app.get("/api/session/{session_id}/stats")
async def get_session_stats(session_id: str):
    """Get session statistics"""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    
    session = sessions[session_id]
    return {
        "session_id": session_id,
        "created_at": session['created_at'].isoformat(),
        "last_activity": session['last_activity'].isoformat(),
        "message_count": session['message_count'],
        "conversation_length": len(rag_system.memory.history) if rag_system else 0
    }


@app.get("/api/suggestions")
async def get_suggestions():
    """Get suggested queries"""
    return {
        "suggestions": [
            {
                "icon": "summarize",
                "text": "Kesavananda Bharati case",
                "query": "What is the Kesavananda Bharati case about?"
            },
            {
                "icon": "gavel",
                "text": "Bail under BNSS",
                "query": "What are the bail provisions under BNSS 2023?"
            },
            {
                "icon": "security",
                "text": "Article 21",
                "query": "Explain Article 21 of the Constitution"
            },
            {
                "icon": "update",
                "text": "New criminal codes",
                "query": "What changed with BNS, BNSS, and BSA?"
            }
        ]
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global error handler"""
    print(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
