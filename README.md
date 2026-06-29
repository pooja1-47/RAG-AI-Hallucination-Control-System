# RAG-AI Hallucination Control System

## Project Overview
A RAG-based verified AI response system that reduces 
LLM hallucinations using FAISS vector search and SQLite caching.

## Developer
- Pooja (Full Stack - Backend + Frontend + RAG Pipeline)

## Technologies Used
- Python, Flask
- FAISS (Vector Search)
- SQLite (Caching)
- Gemini API
- Sentence Transformers
- HTML, CSS, JavaScript

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Build FAISS index: `python scripts/build_index.py`
3. Start Flask: `python backend/app.py`
4. Open `frontend/index.html` in browser