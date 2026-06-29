import os
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

KNOWLEDGE_BASE_DIR = "knowledge_base"
OUTPUT_DIR = "faiss_index"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

def load_txt_docs(folder):
    texts = []
    for f in Path(folder).glob("*.txt"):
        print(f"  Loading: {f.name}")
        texts.append(f.read_text(encoding="utf-8"))
    return texts

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    step = size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i+size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def build_index():
    print("\n=== RAG Index Builder ===\n")
    print(f"Loading documents from '{KNOWLEDGE_BASE_DIR}/'...")
    all_texts = load_txt_docs(KNOWLEDGE_BASE_DIR)

    if not all_texts:
        print("\n[ERROR] No documents found in knowledge_base/")
        return

    print(f"  {len(all_texts)} document(s) loaded.\n")
    print("Chunking documents...")
    chunks = []
    for text in all_texts:
        chunks.extend(chunk_text(text))
    print(f"  {len(chunks)} chunks created.\n")

    print(f"Loading embedding model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    print(f"  Embeddings shape: {embeddings.shape}\n")

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"  Index contains {index.ntotal} vectors.\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    index_path = os.path.join(OUTPUT_DIR, "index.faiss")
    chunks_path = os.path.join(OUTPUT_DIR, "chunks.json")

    faiss.write_index(index, index_path)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved index  → {index_path}")
    print(f"Saved chunks → {chunks_path}")
    print(f"\n✅ Done! Indexed {len(chunks)} chunks from {len(all_texts)} document(s).")

if __name__ == "__main__":
    build_index()