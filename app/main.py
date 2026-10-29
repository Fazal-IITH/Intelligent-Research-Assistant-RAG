from fastapi import FastAPI ,File, UploadFile
from pydantic import BaseModel
index = None 
chunks = None

app = FastAPI()

from app.services.pdf_processor import make_chunks
from app.services.embedding_service import create_embeddings
from app.services.vector_store import store_vec
from app.services.vector_store import index_search
from app.services.llm_services import generate_answer

class QuestionModel(BaseModel):
    question : str

@app.get("/")
def health_check():
    return {"message":"Yeah im working fine"}

@app.post("/upload_file")
async def upload_file(file : UploadFile = File(...)):
    global chunks,index
    content = await file.read()
    chunks = make_chunks(content)
    embeddings = create_embeddings(chunks)
    index = store_vec(embeddings)
    
    return {
        "chunks": len(chunks),
        "embeddings": embeddings.shape[0]
    }

@app.post("/ask_question")
def ask_question(request: QuestionModel):
    question = request.question 
    q_embeddings = create_embeddings([question])
    if index is None:
        return {
            "error": "Please upload a PDF first."
        }
    indices = index_search(index,q_embeddings)
    retrieved_chunks = [
        chunks[i]
        for i in indices[0]
    ]
    answer = generate_answer(retrieved_chunks,question)

    return {"answer":answer}