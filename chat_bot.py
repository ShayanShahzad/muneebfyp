# backend.py
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DEEPSEEK_API_KEY = "sk-4fd8a910f56540b2a379ead689c650a8"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
EMBEDDINGS_FILE = "embeddings.npy"
INDEX_FILE = "faiss_index.index"
DOCUMENTS_FILE = "documents.json"


class ChatRequest(BaseModel):
    message: str


def initialize_system():
    # Initialize embedding model (this should happen in both cases)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load or create embeddings
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(INDEX_FILE) and os.path.exists(DOCUMENTS_FILE):
        print("Loading precomputed embeddings...")
        embeddings = np.load(EMBEDDINGS_FILE)
        index = faiss.read_index(INDEX_FILE)
        with open(DOCUMENTS_FILE, 'r', encoding='utf-8') as f:
            documents = json.load(f)
    else:
        print("Computing embeddings for the first time...")
        with open('nu_all_pages_playwright.json', 'r', encoding='utf-8') as f:
            scraped_data = json.load(f)

        # Preprocess into chunks
        documents = []
        for page in scraped_data:
            content = page['content']
            chunks = [content[i:i + 500] for i in range(0, len(content), 500)]
            for chunk in chunks:
                documents.append({'text': chunk, 'url': page['url']})

        # Create and save embeddings
        embeddings = embedding_model.encode([doc['text'] for doc in documents])
        np.save(EMBEDDINGS_FILE, embeddings)

        # Create and save FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(index, INDEX_FILE)

        # Save documents
        with open(DOCUMENTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)

    return embedding_model, index, documents

# Initialize system at startup
embedding_model, index, documents = initialize_system()


def retrieve_relevant_documents(query, k=3):
    query_embedding = embedding_model.encode(query)
    distances, indices = index.search(np.array([query_embedding]), k)
    return [documents[i] for i in indices[0]]


def generate_with_deepseek_api(query, context_docs):
    context = "\n\n".join([f"📄 Source: {doc['url']}\n{doc['text']}" for doc in context_docs])

    messages = [
        {
            "role": "system",
            "content": "You are an expert assistant for National University (NU) of Pakistan. "
                       "Answer questions based on the provided context. "
                       "If you don't know the answer, say so."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        }
    ]

    response = requests.post(
        DEEPSEEK_API_URL,
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000
        },
        timeout=30
    )

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    return response.json()["choices"][0]["message"]["content"]


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        docs = retrieve_relevant_documents(request.message)
        answer = generate_with_deepseek_api(request.message, docs)

        return {
            "response": answer,
            "sources": [doc['url'] for doc in docs]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)