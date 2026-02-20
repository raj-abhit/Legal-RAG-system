"""
Simplified API Server for Legal RAG System

Provides REST API endpoints for:
- /query: Submit questions and get answers
- /chats: List, create, load, delete previous chats
- /health: System status check

Chat persistence uses SQLite (easy to migrate to MongoDB later).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import os
import sqlite3
import uuid
import logging
from dotenv import load_dotenv

from legal_rag_system import LegalRAGSystem

# Configuration
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_PATH = os.path.join(BASE_DIR, "frontend.html")
DB_PATH = os.getenv("CHAT_DB_PATH", os.path.join(BASE_DIR, "chats.db"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Indian Law Chatbot API",
    description="Legal RAG system with conversation memory",
    version="2.0.0"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
try:
    rag = LegalRAGSystem()
    rag.load_vectorstore()
    logger.info("✓ RAG system loaded successfully")
except Exception as e:
    logger.error(f"✗ Failed to load RAG system: {e}")
    rag = None


# ============================================================================
# Data Models
# ============================================================================

class QueryRequest(BaseModel):
    """Format for incoming query requests."""
    question: str
    chat_id: Optional[str] = None


class QueryResponse(BaseModel):
    """Format for query responses."""
    answer: str
    sources: List[dict]
    chat_id: str
    language: str
    is_followup: bool
    error: bool = False


# ============================================================================
# Chat Storage (SQLite)
# ============================================================================

class ChatStore:
    """SQLite-backed chat persistence. Easy to swap for MongoDB later."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT 'New Chat',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources TEXT DEFAULT '[]',
                    language TEXT DEFAULT '',
                    is_followup INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id)")
        logger.info(f"Chat DB ready: {self.db_path}")

    # ---- Chat CRUD ----

    def create_chat(self, title: str = "New Chat") -> dict:
        chat_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO chats (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (chat_id, title, now, now)
            )
        return {"id": chat_id, "title": title, "created_at": now, "updated_at": now}

    def list_chats(self, limit: int = 50) -> List[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT c.id, c.title, c.created_at, c.updated_at,
                          COUNT(m.id) as message_count
                   FROM chats c LEFT JOIN messages m ON c.id = m.chat_id
                   GROUP BY c.id ORDER BY c.updated_at DESC LIMIT ?""",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_chat(self, chat_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
        return dict(row) if row else None

    def rename_chat(self, chat_id: str, title: str):
        with self._conn() as conn:
            conn.execute("UPDATE chats SET title = ? WHERE id = ?", (title, chat_id))

    def delete_chat(self, chat_id: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))

    def _touch_chat(self, chat_id: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE chats SET updated_at = ? WHERE id = ?",
                (datetime.utcnow().isoformat(), chat_id)
            )

    # ---- Messages ----

    def add_message(self, chat_id: str, question: str, answer: str,
                    sources: str = "[]", language: str = "", is_followup: bool = False):
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO messages (chat_id, question, answer, sources, language, is_followup, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (chat_id, question, answer, sources, language, int(is_followup), now)
            )
        self._touch_chat(chat_id)

    def get_messages(self, chat_id: str) -> List[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE chat_id = ? ORDER BY id ASC",
                (chat_id,)
            ).fetchall()
        return [dict(r) for r in rows]


# Initialize
chat_store = ChatStore()


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Check if system is running and vectorstore is loaded."""
    return {
        "status": "healthy" if rag else "error",
        "timestamp": datetime.utcnow().isoformat(),
        "vectorstore_loaded": rag is not None
    }


# ---- Chat management ----

@app.get("/chats")
async def list_chats():
    """List all previous chats, newest first."""
    return chat_store.list_chats()


@app.post("/chats")
async def create_chat():
    """Create a new empty chat."""
    rag.reset_conversation()
    return chat_store.create_chat()


@app.get("/chats/{chat_id}")
async def get_chat(chat_id: str):
    """Load a previous chat with all its messages."""
    chat = chat_store.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    messages = chat_store.get_messages(chat_id)

    # Rebuild RAG conversation memory from this chat's history
    rag.reset_conversation()
    for msg in messages:
        rag.memory.add_exchange(msg["question"], msg["answer"])

    return {"chat": chat, "messages": messages}


@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete a chat and all its messages."""
    chat_store.delete_chat(chat_id)
    return {"status": "deleted"}


@app.patch("/chats/{chat_id}")
async def rename_chat(chat_id: str, title: str):
    """Rename a chat."""
    chat_store.rename_chat(chat_id, title)
    return {"status": "renamed"}


# ---- Query ----

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Submit a legal question and get an answer.
    Auto-creates a chat if no chat_id provided.
    Auto-titles chat from first question.
    """
    if not rag:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if len(question) > 5000:
        raise HTTPException(status_code=400, detail="Question too long (max 5000 chars)")

    # Get or create chat
    chat_id = request.chat_id
    is_new_chat = False
    if not chat_id or not chat_store.get_chat(chat_id):
        title = question[:60] + ("..." if len(question) > 60 else "")
        chat = chat_store.create_chat(title=title)
        chat_id = chat["id"]
        is_new_chat = True

    try:
        result = rag.query(question)

        import json as _json
        sources_json = _json.dumps(result.get("sources", []))

        chat_store.add_message(
            chat_id, question, result["answer"],
            sources=sources_json,
            language=result.get("language", ""),
            is_followup=result.get("is_followup", False)
        )

        # Auto-title on first message if chat was pre-created empty
        if not is_new_chat:
            chat = chat_store.get_chat(chat_id)
            if chat and chat["title"] == "New Chat":
                chat_store.rename_chat(chat_id, question[:60] + ("..." if len(question) > 60 else ""))

        return QueryResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            chat_id=chat_id,
            language=result.get("language", ""),
            is_followup=result.get("is_followup", False),
            error=result.get("error", False)
        )

    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing query. Please try again.")


@app.get("/")
async def root():
    """Serve web UI."""
    if os.path.exists(FRONTEND_PATH):
        return FileResponse(FRONTEND_PATH)
    return {"message": "Legal RAG API is running", "docs": "/docs"}


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize on startup."""
    logger.info("API Server starting...")
    if rag:
        logger.info("RAG system ready for queries")
    else:
        logger.warning("RAG system not initialized - queries will fail")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("API Server shutting down...")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", "8000"))
    host = os.getenv("API_HOST", "0.0.0.0")
    
    print("\n" + "="*70)
    print("LEGAL RAG API SERVER")
    print("="*70)
    print(f"📍 Starting on {host}:{port}")
    print(f"📚 Docs: http://localhost:{port}/docs")
    print(f"💬 Chat UI: http://localhost:{port}/")
    print("="*70 + "\n")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
