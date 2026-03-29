# RAG vs LlamaIndex: A Practical Comparison Lab

> **A hands-on engineering lab built to answer one question: What does a RAG framework actually do for you?**

This repository contains two complete, working implementations of a Retrieval-Augmented Generation (RAG) pipeline — one built entirely from scratch, and one using the LlamaIndex framework — to provide a true apples-to-apples architectural comparison.

---

## 🧠 What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that gives Large Language Models (LLMs) access to specific, private data — preventing "hallucinations" by grounding the model's response in retrieved context.

**The 4-stage pipeline:**
```
Your Document
     │
     ▼
[1. CHUNK]  → Split text into overlapping windows
     │
     ▼
[2. EMBED]  → Convert each chunk into a vector (a list of numbers)
     │
     ▼
[3. RETRIEVE] → Find the chunks mathematically closest to the question
     │
     ▼
[4. GENERATE] → Feed chunks + question to an LLM for a final answer
```

---

## 📁 Project Structure

```
rag-vs-llamaindex-lab/
│
├── pure_rag/                      # Manual, from-scratch implementation
│   ├── 01_basic_pipeline.ipynb    # End-to-end manual RAG
│   ├── chunking_utils.py          # Custom sliding window chunker
│   └── retrieval_utils.py         # Custom dot-product retriever
│
├── llamaindex_rag/                # Framework-powered implementation
│   ├── 01_basic_pipeline.ipynb    # Default LlamaIndex RAG (5 lines!)
│   ├── 02_custom_chunking.ipynb   # SentenceSplitter + transformations
│   └── 03_retrieval_and_query.ipynb # Retrievers, top_k, source nodes
│
├── comparisons/
│   └── retrieval_quality.md       # Full side-by-side analysis
│
├── data/                          # Sample datasets
└── utils/                         # Helper scripts
```

---

## ⚔️ The Comparison at a Glance

| Stage | Pure RAG | LlamaIndex | Winner |
|:---|:---|:---|:---:|
| **Chunking** | Character-based sliding window | Token-aware `SentenceSplitter` | 🤝 Tie |
| **Embedding** | Manual loop + API calls | `Settings.embed_model` + auto-batch | ⚡ LlamaIndex |
| **Indexing** | Python list + NumPy | `VectorStoreIndex` | ⚡ LlamaIndex |
| **Retrieval** | Manual dot-product sort | `.as_retriever(similarity_top_k=N)` | ⚡ LlamaIndex |
| **Synthesis** | Manual prompt template | `CompactAndRefine` auto-synthesizer | ⚡ LlamaIndex |
| **Transparency** | Every line is visible | Need `source_nodes` to see under the hood | 🔬 Pure RAG |
| **Total Code** | ~115 lines | ~7 lines | ⚡ LlamaIndex |

---

## 🔑 Key Learnings

### 1. `similarity_top_k` is everything
In our **Vault Experiment** (`03_retrieval_and_query.ipynb`), the model failed to answer "What is inside the vault?" with `top_k=2` — because the golden bar fact was ranked 3rd. With `top_k=3`, it answered perfectly. **Always calibrate your top-k to your use case.**

### 2. The LLM never touches retrieval
The Embedding Model (`gemini-embedding-001`) handles the search math. The LLM (`gemini-2.0-flash`) only activates at the **final generation step**. This makes retrieval fast and cheap.

### 3. Token-aware chunking > Character chunking
LlamaIndex's `SentenceSplitter` won't cut a word in half. Our manual character chunker might. In production, always use boundary-aware chunking.

### 4. Frameworks hide the math, not replace it
LlamaIndex runs the **exact same dot-product math** under the hood. It just wraps it in a friendly API. Understanding Pure RAG first means you can debug LlamaIndex when it misbehaves.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|:---|:---|
| `Python 3.12` | Core language |
| `LlamaIndex` | RAG Framework |
| `llama-index-llms-google-genai` | Gemini LLM integration |
| `llama-index-embeddings-google-genai` | Gemini Embedding integration |
| `google-generativeai` | Direct Gemini API (Pure RAG) |
| `nest_asyncio` | Notebook async compatibility fix |
| `python-dotenv` | API key management |
| `uv` | Fast Python package manager |

**Models Used:**
- 🧠 **LLM**: `models/gemini-2.0-flash`
- 🔢 **Embeddings**: `models/gemini-embedding-001`

---

## 🚀 Setup & Running

```bash
# 1. Clone the repository
git clone https://github.com/Edge-Explorer/rag-vs-llamaindex-lab.git
cd rag-vs-llamaindex-lab

# 2. Install dependencies (using uv)
uv sync

# 3. Add your API key
echo "GEMINI_API_KEY=your_key_here" > .env

# 4. Start Jupyter
uv run jupyter notebook
```

> 💡 **Note**: Get your free Gemini API key at [aistudio.google.com](https://aistudio.google.com)

---

## 📖 Recommended Learning Order

1. `pure_rag/01_basic_pipeline.ipynb` → Understand the mechanics.
2. `llamaindex_rag/01_basic_pipeline.ipynb` → See the abstraction.
3. `llamaindex_rag/02_custom_chunking.ipynb` → Control the splitter.
4. `llamaindex_rag/03_retrieval_and_query.ipynb` → Master top-k & source nodes.
5. `comparisons/retrieval_quality.md` → Read the full verdict.

---

## 🏗️ Architecture Decision Guide

```
When should you use each approach?

Pure RAG  →  Learning, Research, Maximum Control
LlamaIndex →  Prototyping, Production, Team Projects
```

---

*Built as part of a structured AI Engineering curriculum to deeply understand RAG — not just use it.*
