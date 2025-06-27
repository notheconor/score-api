import os
import json
import numpy as np
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load stored embeddings
with open("score_embeddings.json", "r") as f:
    documents = json.load(f)

EMBEDDING_MODEL = "text-embedding-3-small"

def get_embedding(text, model=EMBEDDING_MODEL):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def expand_query(user_query):
    prompt = f"""You are a surgical resident preparing for the ABSITE exam.
Interpret the following query and expand it to include relevant clinical terminology,
synonyms, eponyms, anatomic locations, and procedures that would help locate it in a
surgical textbook or clinical reference. Limit to 10.

Query: "{user_query}"

Expanded Search Terms:"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()

def load_full_module_text(file_name):
    path = os.path.join("SCORE Modules", file_name)
    with open(path, "r") as f:
        return f.read()

def search_score_modules(user_query, top_k=5):
    # Step 1: Expand query
    expanded_query = expand_query(user_query)
    print(f"\nExpanded Query:\n{expanded_query}\n")

    # Step 2: Get embedding of expanded query
    query_embedding = get_embedding(expanded_query)

    # Step 3: Rank all chunks by similarity
    ranked_chunks = []
    for doc in documents:
        similarity = cosine_similarity(query_embedding, doc["embedding"])
        ranked_chunks.append((similarity, doc))

    # Step 4: Retrieve top full modules (deduplicated by file)
    seen_files = set()
    top_full_modules = []

    for score, doc in sorted(ranked_chunks, key=lambda x: x[0], reverse=True):
        file_name = doc["file"]
        if file_name not in seen_files:
            full_text = load_full_module_text(file_name)
            top_full_modules.append((score, file_name, full_text))
            seen_files.add(file_name)
        if len(top_full_modules) >= top_k:
            break

    return top_full_modules

if __name__ == "__main__":
    query = input("Enter your clinical question: ")
    results = search_score_modules(query)

    for i, (score, file_name, full_text) in enumerate(results, 1):
        print(f"\n--- Result {i} ({file_name}, Score: {score:.3f}) ---\n{full_text}...\n")
