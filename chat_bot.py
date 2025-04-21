import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import requests

# Load scraped data
with open('nu_all_pages_playwright.json', 'r', encoding='utf-8') as f:
    scraped_data = json.load(f)

# Preprocess into chunks
documents = []
for page in scraped_data:
    content = page['content']
    chunks = [content[i:i + 500] for i in range(0, len(content), 500)]
    for chunk in chunks:
        documents.append({'text': chunk, 'url': page['url']})

# Create embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding_model.encode([doc['text'] for doc in documents])

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# DeepSeek API
DEEPSEEK_API_KEY = "sk-4fd8a910f56540b2a379ead689c650a8"  # 🔑 Replace with your key
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"


def retrieve_relevant_documents(query, k=3):
    query_embedding = embedding_model.encode(query)
    distances, indices = index.search(np.array([query_embedding]), k)
    return [documents[i] for i in indices[0]]


def generate_with_deepseek_api(query, context_docs):
    context = "\n\n".join([f"📄 Source: {doc['url']}\n{doc['text']}" for doc in context_docs])

    messages = [
        {"role": "system", "content": "You are an NU Pakistan assistant. Answer based on the context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"}
    ]

    response = requests.post(
        DEEPSEEK_API_URL,
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
        json={"model": "deepseek-chat", "messages": messages, "temperature": 0.7}
    )

    return response.json()["choices"][0]["message"]["content"]


def chat():
    print("🔵 NU Pakistan Chatbot - Type 'exit' to quit")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break

        docs = retrieve_relevant_documents(user_input)
        answer = generate_with_deepseek_api(user_input, docs)

        print(f"\n🤖 Assistant: {answer}")
        print("\n📚 Sources:")
        for doc in docs:
            print(f"- {doc['url']}")


if __name__ == "__main__":
    chat()