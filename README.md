# RAG vs LlamaIndex: A Practical Comparison Lab

This repository explores and compares two approaches to building Retrieval-Augmented Generation (RAG) systems:

1. **Pure RAG**: A from-scratch implementation where the entire pipeline is manually built, focusing on understanding the core mechanics (chunking, embedding, vector search, and completion).
2. **LlamaIndex RAG**: An implementation leveraging the LlamaIndex framework to understand how its abstractions simplify complex workflows.

## Project Structure

- `pure_rag/`: From-scratch implementation of the RAG pipeline.
- `llamaindex_rag/`: Implementation using the LlamaIndex framework.
- `comparisons/`: Documentation and analysis comparing the two approaches.
- `data/`: Sample datasets for testing.
- `utils/`: Helper scripts for data processing or evaluation.

## Key Questions Explored

- What does LlamaIndex abstract away?
- When should we use pure RAG vs frameworks?
- What are the performance differences (latency, accuracy)?
- How does retrieval quality compare between manual and automated indexing?

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```
2. Configure environment variables in `.env`.
3. Follow the guides in each folder to begin the comparison.
