from backend.pdf_reader import extract_text
from backend.embeddings import create_embeddings
from backend.vector_store import create_vector_store

def process_pdf(pdf_path):
    text = extract_text(pdf_path)

    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    embeddings = create_embeddings(chunks)

    index = create_vector_store(embeddings)

    return chunks, index