# 03 Retrieval (Pure RAG)

import os 
import numpy as np

class SimpleRetriever:
    def __init__(self, embeddings, chunks):
        self.embeddings= embeddings
        self.chunks= chunks
    
    def search(self, query_embedding, k=1):
        # 1. Math: Convert to NumPy arrays for dot product
        emb_array = np.array(self.embeddings)
        q_array = np.array(query_embedding)

        # 2. Math: Dot Product
        similarities = np.dot(emb_array, q_array)

        # 2. Find the index of the highest score
        idx= np.argmax(similarities)

        # 3. Return the text associated with that math
        return self.chunks[idx]