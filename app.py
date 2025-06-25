from flask import Flask, request, jsonify
import openai
import json
import numpy as np
import os

# === CONFIG ===
openai.api_key = os.getenv("OPENAI_API_KEY")  # Set this in your hosting environment
EMBEDDINGS_FILE = "score_embeddings.json"
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_N_RESULTS = 3
# ==============

app = Flask(__name__)

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(text):
    response = openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

def load_embeddings():
    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

@app.route("/search", methods=["POST"])
def search_score():
    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"error": "Missing 'question' field"}), 400

    user_embedding = get_embedding(question)
    chunks = load_embeddings()

    scored = []
    for item in chunks:
        sim = cosine_similarity(user_embedding, item["embedding"])
        scored.append((sim, item))

    top = sorted(scored, reverse=True, key=lambda x: x[0])[:TOP_N_RESULTS]
    results = [{
        "file": item["file"],
        "score": float(sim),
        "text": item["text"]
    } for sim, item in top]

    return jsonify({"results": results})

@app.route("/")
def health():
    return "✅ SCORE Search API is live!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

