# Latency Analysis: Pure RAG vs LlamaIndex

**Real benchmark results measured on 2026-03-29 using `time.time()` in `latency/00_latency_benchmark.ipynb`.**

---

## 🏆 Official Results

| Pipeline Stage | Pure RAG (Manual) | LlamaIndex (Framework) |
|:---|:---:|:---:|
| **Chunking / Indexing** | `0.0000s` | `0.7142s` (chunk + embed) |
| **Embedding** | `9.6219s` (serial loop) | *(included above)* |
| **Retrieval** | `0.4761s` | `0.4777s` |
| **Generation** | `1.0453s` | `1.5026s` |
| **TOTAL** | **`11.1433s`** | **`2.6945s`** |

> 🥇 **LlamaIndex is 4.1x faster** than Pure RAG on the same hardware, same API, same data.

---

## 🕵️ Root Cause Analysis

### Why did Pure RAG lose by so much?

The bottleneck was **serial embedding** — the `for` loop that sends one API call at a time:

```
Pure RAG Embedding Loop:
Chunk 1 → [send] → [wait] → [receive]    (0.6s)
Chunk 2 → [send] → [wait] → [receive]    (0.6s)
Chunk 3 → [send] → [wait] → [receive]    (0.6s)
...×16 chunks
= 9.62 seconds total
```

### Why did LlamaIndex win?

LlamaIndex sends all chunks **in a single batched API request**, eliminating the round-trip wait for each chunk:

```
LlamaIndex Batched Embedding:
[Chunk 1, Chunk 2, ..., Chunk 16] → [send once] → [receive all]
= 0.71 seconds total (including chunking!)
```

**This is a 13.5x improvement on the embedding step alone.**

---

## 📊 Stage-by-Stage Breakdown

### Stage 1: Chunking

| | Pure RAG | LlamaIndex |
|:---|:---:|:---:|
| Time | `0.0000s` | Combined with Embedding |
| Method | Character sliding window | Token-aware `SentenceSplitter` |
| Notes | Nearly instant (pure Python) | — |

**Key Insight:** Chunking itself is essentially **free**. It's all local CPU math — no network, no API. This confirms that chunking strategy choice is about *quality*, not speed.

---

### Stage 2: Embedding

| | Pure RAG | LlamaIndex |
|:---|:---:|:---:|
| Time | `9.6219s` | `~0.71s` *(batched)* |
| Pattern | Serial `for` loop | Internal batch request |
| API Calls | 16 separate calls | 1 batched call |

**Key Insight:** The embedding stage was responsible for **86% of Pure RAG's total time**. This is THE bottleneck in any RAG system — and batching eliminates it.

---

### Stage 3: Retrieval

| | Pure RAG | LlamaIndex |
|:---|:---:|:---:|
| Time | `0.4761s` | `0.4777s` |
| Method | NumPy dot product | Cosine similarity (internal) |
| API Calls | 1 *(query embedding)* | 1 *(query embedding)* |

**Key Insight:** Both pipelines were **identical** here — both had to embed the query (one API call) and then do vector math. No framework advantage.

---

### Stage 4: Generation

| | Pure RAG | LlamaIndex |
|:---|:---:|:---:|
| Time | `1.0453s` | `1.5026s` |
| Method | Manual prompt template | `CompactAndRefine` synthesizer |
| API Calls | 1 | 1+ *(may re-rank/refine)* |

**Key Insight:** LlamaIndex was *slightly slower* here because `CompactAndRefine` synthesizer may make additional internal calls to structure the response. For simple queries, a manual prompt is faster.

---

## 🔑 Key Takeaways

1. **Batching wins.** The single biggest optimization any RAG system can make is to stop sending embeddings one by one.

2. **Retrieval is always near-instant.** Vector similarity math takes milliseconds regardless of approach.

3. **Generation dominates at scale.** For complex multi-document queries, LLM generation time grows. For simple RAG, it's ~1s.

4. **Frameworks trade transparency for efficiency.** LlamaIndex's "magic" is engineering you don't have to write yourself.

---

## 📋 Test Parameters

| Parameter | Value |
|:---|:---|
| **Test text** | "LlamaIndex is a data framework..." × 30 repetitions |
| **Text length** | 3,060 characters |
| **Total chunks (Pure RAG)** | 16 chunks (`size=200, overlap=50`) |
| **Query** | "What is LlamaIndex?" |
| **Embedding model** | `models/gemini-embedding-001` |
| **LLM** | `models/gemini-2.0-flash` |
| **Benchmark date** | 2026-03-29 |
