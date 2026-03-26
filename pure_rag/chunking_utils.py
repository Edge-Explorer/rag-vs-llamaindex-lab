# 02 Chunking Comparison (Pure RAG)
import os

def sliding_window_chunker(text, size=500, overlap=100):
    chunks = []
    # Loop through the text, but jump back by the `overlap` every time!
    for i in range(0, len(text), size - overlap):
        chunk = text[i : i + size]
        chunks.append(chunk)
    return chunks