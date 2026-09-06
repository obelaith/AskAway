from fastapi import FastAPI
from pydantic import BaseModel

from askaway.services.rag_service import RAGService

app = FastAPI(
    title="AskAway API",
    description="Retrieval Augmented Generation API",
    version="1.0"
)



from pathlib import Path

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]

CORPUS_PATH = r"C:\Users\laith\Desktop\Portfolio\AskAway\data\processed\chunks.jsonl"

rag_service = RAGService(
    CORPUS_PATH
)



class QuestionRequest(BaseModel):

    question: str



class QuestionResponse(BaseModel):

    answer: str
    sources: list



@app.get("/")
def health():

    return {
        "status": "AskAway API running"
    }



@app.post(
    "/ask",
    response_model=QuestionResponse
)
def ask_question(
    request: QuestionRequest
):

    result = rag_service.ask(
        request.question
    )


    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }