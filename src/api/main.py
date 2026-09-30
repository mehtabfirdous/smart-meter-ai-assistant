from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.generation.rag_pipeline import generate_answer


app = FastAPI(
    title="Smart Meter AI Assistant",
    description="RAG-based AI assistant for smart meter operations",
    version="1.0.0"
)


# ---------------------------------------------------------
# CORS CONFIGURATION
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# REQUEST MODEL
# ---------------------------------------------------------

class QuestionRequest(BaseModel):

    question: str

    top_k: int = 3


# ---------------------------------------------------------
# ROOT ENDPOINT
# ---------------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "Smart Meter AI Assistant API is running",
        "status": "ok"
    }


# ---------------------------------------------------------
# HEALTH ENDPOINT
# ---------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ---------------------------------------------------------
# ASK ENDPOINT
# ---------------------------------------------------------

@app.post("/ask")
def ask_question(request: QuestionRequest):

    result = generate_answer(
        request.question,
        top_k=request.top_k
    )

    sources = []

    for source in result["sources"]:

        metadata = source["metadata"]

        sources.append({
            "source": metadata["source"],
            "page": metadata["page"],
            "score": source["score"]
        })

    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": sources
    }