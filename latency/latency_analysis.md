# Latency Analysis: Pure RAG vs LlamaIndex

A breakdown of where time is spent in each pipeline stage, and what we observed during the lab.

---

> ⚠️ **Honest Note:** We did not run formal benchmarks with `time.time()` in this lab. The observations below are **qualitative** — based on what we directly experienced while running both pipelines. The "Next Steps" section provides the exact code to run a real benchmark.

---

## 1. Where Time Is Spent in a RAG Pipeline

```
Stage 1: CHUNKING       → Fast (CPU only, no API)
Stage 2: EMBEDDING      → Slow (API call per chunk or batch)
Stage 3: INDEXING       → Medium (storing vectors in memory)
Stage 4: RETRIEVAL      → Very Fast (dot-product math, no API)
Stage 5: GENERATION     → Slowest (full LLM inference)
```

**The Golden Rule:** Embedding + Generation dominate latency. Chunking and Retrieval are nearly instant.

---

## 2. Qualitative Observations from the Lab

### Embedding Phase (Both approaches use `gemini-embedding-001`)

| Observation | Pure RAG | LlamaIndex |
|:---|:---|:---|
| API calls | 1 per chunk (manual loop) | Batched automatically |
| Perceived speed | Slower for many chunks | Noticeably faster for same input |
| Network round-trips | N (one per chunk) | Fewer (batched) |

**Why LlamaIndex felt faster during embedding:** It batches multiple chunks into a single API call instead of making N separate requests. This reduces network overhead significantly.

### Retrieval Phase (Pure dot-product math — no API)

| Observation | Pure RAG | LlamaIndex |
|:---|:---|:---|
| Algorithm | NumPy dot product | Cosine similarity (internal) |
| Perceived speed | Instant | Instant |
| API calls | ❌ None | ❌ None |

Both were near-instant. **Retrieval is cheap.** The LLM is never involved here — it's pure math.

### Generation Phase (Both use `gemini-2.0-flash`)

| Observation | Pure RAG | LlamaIndex |
|:---|:---|:---|
| API calls | 1 (manual prompt) | 1 (`CompactAndRefine`) |
| Perceived speed | Similar | Similar |
| Tokens sent | Manual — we control it | Automatic — may send more context |

Both pipelines had similar generation speed since they use the same model. LlamaIndex may send slightly more tokens depending on how `CompactAndRefine` formats the context.

---

## 3. Theoretical Latency Breakdown

```
Total Latency = Chunking + (Embedding × N chunks) + Retrieval + Generation

Pure RAG:    ≈ 5ms   + (250ms × N)   + 2ms  + 1200ms
LlamaIndex:  ≈ 5ms   + (150ms × N*)  + 2ms  + 1200ms
                              ↑
                    *Batching reduces per-chunk embedding time
```

**Estimated savings from LlamaIndex batching:** ~40% faster during indexing for large document sets.

---

## 4. When Latency Actually Matters

| Scale | Pure RAG Impact | LlamaIndex Impact |
|:---|:---|:---|
| 1–10 chunks (our lab) | ✅ Negligible | ✅ Negligible |
| 100–1,000 chunks | ⚠️ Embedding loop becomes slow | ✅ Batching helps significantly |
| 10,000+ chunks | ❌ Manual embedding is a bottleneck | ✅ Async + batching essential |

For our lab (3–10 chunks), both approaches felt identical. At production scale, LlamaIndex's batching and async support make a real difference.

---

## 5. How to Run a Real Benchmark (Next Steps)

Add this to any notebook to measure exact timings:

```python
import time

# Benchmark Embedding
start = time.time()
index = VectorStoreIndex.from_documents(documents)
embed_time = time.time() - start
print(f"Indexing Time: {embed_time:.3f}s")

# Benchmark Retrieval
start = time.time()
results = retriever.retrieve("Your question here")
retrieve_time = time.time() - start
print(f"Retrieval Time: {retrieve_time:.4f}s")

# Benchmark Generation
start = time.time()
response = query_engine.query("Your question here")
gen_time = time.time() - start
print(f"Generation Time: {gen_time:.3f}s")

print(f"\nTotal Pipeline: {embed_time + retrieve_time + gen_time:.3f}s")
```

---

## 6. Key Takeaways

1. **Retrieval is always fast** — it's dot-product math with zero API calls.
2. **Generation is always the bottleneck** — the LLM is the slow part.
3. **LlamaIndex wins on embedding speed** at scale due to batching.
4. **For small documents (our lab)**: difference is negligible.
5. **Formal benchmarking** is a planned next step for this project.
