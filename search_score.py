import os
import json
import numpy as np
from openai import OpenAI

# Set your API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # or hardcode your key here

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Load stored embeddings
with open("score_embeddings.json", "r") as f:
    documents = json.load(f)

EMBEDDING_MODEL = "text-embedding-3-small"

def expand_query(user_query):
    prompt = f"""You are a surgical resident preparing for the ABSITE exam.
Interpret the following query and expand it to include relevant clinical terminology,
synonyms, eponyms, anatomic locations, and procedures that would help locate it in a
surgical textbook or clinical reference.

Query: "{user_query}"

Expanded Search Terms:"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    expanded = response.choices[0].message.content.strip()
    return expanded

def search_score_modules(user_query, top_k=5):
    expanded_query = expand_query(user_query)
    print(f"\nExpanded Query:\n{expanded_query}\n")

    query_embedding = get_embedding(expanded_query, model=EMBEDDING_MODEL)

    ranked_chunks = []
    for doc in documents:
        chunk_embedding = doc["embedding"]
        similarity = cosine_similarity(query_embedding, chunk_embedding)
        ranked_chunks.append((similarity, doc))

    top_chunks = sorted(ranked_chunks, key=lambda x: x[0], reverse=True)[:top_k]
    return [(score, doc["text"]) for score, doc in top_chunks]

if __name__ == "__main__":
    query = input("Enter your clinical question: ")
    results = search_score_modules(query)

    for i, (score, text) in enumerate(results, 1):
        print(f"\n--- Result {i} (Score: {score:.3f}) ---\n{text}\n")
