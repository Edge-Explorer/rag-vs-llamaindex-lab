# 03 Retrieval (Pure RAG)

import os 
import numpy as np

class SimpleRetriever:
    def __init__(self, embeddings, chunks):
        self.embeddings= embeddings
        self.chunks= chunks
    
    def search(self, query_embedding, k=1):
        # 1. Math: Dot Product (since our model is already normalized)
        similarities= np.dot(self.embeddings, query_embedding.T)

        # 2. Find the index of the highest score
        idx= np.argmax(similarities)

        # 3. Return the text associated with that math
        return self.chunks[idx]