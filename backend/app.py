
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import hashlib
import sqlite3
import json

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

load_dotenv()

app = Flask(__name__)
CORS(app)

# ── Configuration ─────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DB_PATH = "answers.db"
FAISS_INDEX_PATH = "../faiss_index/index.faiss"
CHUNKS_PATH = "../faiss_index/chunks.json"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Load FAISS index and embedding model ─────────────────
print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"Loaded {len(chunks)} chunks.")

# ── SQLite Cache Setup ─────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS answers (
            question_hash TEXT PRIMARY KEY,
            answer TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    return conn

def hash_question(question):
    return hashlib.sha256(question.strip().lower().encode()).hexdigest()

def get_cached(q_hash):
    conn = get_db()
    row = conn.execute(
        "SELECT answer, created_at FROM answers WHERE question_hash=?",
        (q_hash,)
    ).fetchone()
    conn.close()
    if row:
        return {"found": True, "answer": row[0], "timestamp": row[1]}
    return {"found": False}

def store_cached(q_hash, answer):
    conn = get_db()
    conn.execute(
        "INSERT OR IGNORE INTO answers (question_hash, answer) VALUES (?, ?)",
        (q_hash, answer)
    )
    conn.commit()
    conn.close()

# ── FAISS Retrieval ──────────────────────────────────────
def retrieve_context(question, top_k=3):
    q_vec = embed_model.encode([question]).astype("float32")
    distances, indices = index.search(q_vec, top_k)
    results = [chunks[i] for i in indices[0] if i < len(chunks)]
    return "\n\n".join(results)

# ── Routes ────────────────────────────────────────────────
@app.route("/")
def home():
    return jsonify({"message": "RAG AI System is running!"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "chunks_loaded": len(chunks)})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Question is required"}), 400

    # Step 1: Check cache
    q_hash = hash_question(question)
    cached = get_cached(q_hash)
    if cached["found"]:
        return jsonify({
            "answer": cached["answer"],
            "source": "cache",
            "timestamp": cached["timestamp"]
        })

    # Step 2: Retrieve context from FAISS
    context = retrieve_context(question)

    # Step 3: Build prompt and call Gemini
    prompt = f"""Answer ONLY using the context below. 
If the answer is not in the context, say "I cannot find this in the knowledge base."

CONTEXT:
{context}

QUESTION: {question}

Give a clear, factual answer."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
    )
        answer = response.choices[0].message.content.strip()
    except Exception as e: 
        
        return jsonify({"error": f"AI error: {str(e)}"}), 500
    # Step 4: Cache the answer
    store_cached(q_hash, answer)

    return jsonify({"answer": answer, "source": "ai"})

if __name__ == "__main__":
    print("\n🚀 RAG AI System running on http://localhost:5000\n")
    app.run(debug=True, port=5000)