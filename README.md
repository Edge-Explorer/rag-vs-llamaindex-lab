# RAG vs LlamaIndex: A Practical Comparison Lab

> **A hands-on engineering lab built to answer one question: What does a RAG framework actually do for you?**

This repository contains two complete, working implementations of a Retrieval-Augmented Generation (RAG) pipeline — one built entirely from scratch, and one using the LlamaIndex framework — to provide a true apples-to-apples architectural comparison, including a **real-time latency benchmark**.

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

## 🏆 Benchmark Results (Real Data)

> Measured on 2026-03-29. See `latency/00_latency_benchmark.ipynb` for full details.

| Stage | Pure RAG | LlamaIndex | Winner |
|:---|:---:|:---:|:---:|
| **Chunking** | `0.0000s` | *(combined)* | 🤝 |
| **Embedding** | `9.6219s` | `0.7142s` | ⚡ LlamaIndex |
| **Retrieval** | `0.4761s` | `0.4777s` | 🤝 Tie |
| **Generation** | `1.0453s` | `1.5026s` | 🔬 Pure RAG |
| **TOTAL** | **`11.14s`** | **`2.69s`** | ⚡ **LlamaIndex (4.1x faster)** |

**The verdict:** LlamaIndex's **batch embedding** eliminated the serial API loop bottleneck — reducing embedding time from 9.6s to 0.7s (13.5x faster on that stage alone).

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
├── latency/                       # Real-world latency measurements
│   ├── 00_latency_benchmark.ipynb # The stopwatch notebook
│   └── latency_analysis.md        # Full analysis with real numbers
│
├── comparisons/
│   ├── retrieval_quality.md       # Stage-by-stage retrieval breakdown
│   └── chunking_comparison.md     # Character vs token chunking analysis
│
└── data/                          # Sample datasets
```

---

## ⚔️ The Architecture Comparison

| Dimension | Pure RAG | LlamaIndex |
|:---|:---|:---|
| **Chunking** | Character sliding window | Token-aware `SentenceSplitter` |
| **Embedding** | Serial `for` loop (slow) | Batched requests (fast) |
| **Indexing** | Python list + NumPy | `VectorStoreIndex` |
| **Retrieval** | Manual dot-product | `.as_retriever(similarity_top_k=N)` |
| **Synthesis** | Manual prompt template | `CompactAndRefine` |
| **Total Code** | ~115 lines | ~7 lines |
| **Transparency** | ✅ Every line visible | ⚠️ Opt-in via `source_nodes` |

---

## 🔑 Key Learnings

### 1. Batching is the #1 optimization
Going from a `for` loop to batched embedding requests was a **13.5x speedup** on that single stage. This is the most important engineering lesson in this entire lab.

### 2. `similarity_top_k` is everything
In our **Vault Experiment** (`03_retrieval_and_query.ipynb`), the model failed to answer with `top_k=2` — the fact was ranked 3rd. With `top_k=3`, it succeeded. Always calibrate your top-k.

### 3. The LLM never touches retrieval
The Embedding Model (`gemini-embedding-001`) handles ALL the search math. The LLM (`gemini-2.0-flash`) only activates at the **final generation step**. Retrieval is fast and cheap.

### 4. Token-aware chunking > Character chunking
`SentenceSplitter` respects sentence boundaries. Our character chunker can cut a word in half. In production, always use boundary-aware chunking.

### 5. Frameworks hide the math — they don't replace it
LlamaIndex runs the **exact same dot-product math** under the hood. Understanding Pure RAG first means you can debug LlamaIndex when it misbehaves.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|:---|:---|
| `Python 3.12` | Core language |
| `LlamaIndex` | RAG Framework |
| `llama-index-llms-google-genai` | Gemini LLM integration |
| `llama-index-embeddings-google-genai` | Gemini Embedding integration |
| `google-generativeai` | Direct Gemini API (Pure RAG) |
| `nest_asyncio` | Notebook async compatibility |
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

> 💡 Get your free Gemini API key at [aistudio.google.com](https://aistudio.google.com)

---

## 📖 Recommended Learning Order

1. `pure_rag/01_basic_pipeline.ipynb` → Understand the mechanics.
2. `llamaindex_rag/01_basic_pipeline.ipynb` → See the abstraction.
3. `llamaindex_rag/02_custom_chunking.ipynb` → Control the splitter.
4. `llamaindex_rag/03_retrieval_and_query.ipynb` → Master top-k & source nodes.
5. `latency/00_latency_benchmark.ipynb` → **Run the stopwatch yourself.**
6. `comparisons/` → Read the full verdicts.

---

## 🏗️ Architecture Decision Guide

```
When should you use each approach?

Pure RAG  →  Learning, Research, Maximum Control, Debugging
LlamaIndex →  Prototyping, Production, Team Projects, Scale
```

---

*Built as part of a structured AI Engineering curriculum to deeply understand RAG — not just use it.*
