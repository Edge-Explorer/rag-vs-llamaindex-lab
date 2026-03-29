# Retrieval Quality Analysis: Pure RAG vs LlamaIndex

A detailed side-by-side breakdown of what we built, what we learned, and what the data showed.

---

## 1. Chunking (Text Splitting)

| Dimension | Pure RAG (Manual) | LlamaIndex (`SentenceSplitter`) |
|:---|:---|:---|
| **Unit** | Characters | Tokens (~4 chars each) |
| **Boundary Safety** | ❌ Can cut mid-word | ✅ Respects sentence boundaries |
| **Overlap** | Manual sliding logic | `chunk_overlap` parameter |
| **Control** | Total — every line is ours | High — single parameter change |
| **Lines of Code** | ~40 lines (`chunking_utils.py`) | 3 lines |

**Key Insight:** Our Pure RAG splitter could slice `"Generation"` into `"Genera-"` and `"-tion"` mid-word. LlamaIndex's `SentenceSplitter` reads like a human editor — it finishes the sentence before cutting.

---

## 2. Embedding

| Dimension | Pure RAG (Manual) | LlamaIndex (`GoogleGenAIEmbedding`) |
|:---|:---|:---|
| **Implementation** | Direct `google.generativeai` calls | Configured once via `Settings.embed_model` |
| **Batching** | Manual loop per chunk | Handled automatically |
| **Storage** | Plain Python list | Managed internally by `VectorStoreIndex` |
| **Model Used** | `models/gemini-embedding-001` | `models/gemini-embedding-001` |

**Key Insight:** Same model, same math — but LlamaIndex removes the boilerplate of looping over chunks manually and storing results.

---

## 3. Retrieval (Similarity Search)

| Dimension | Pure RAG (Manual) | LlamaIndex (`VectorIndexRetriever`) |
|:---|:---|:---|
| **Algorithm** | Manual dot-product (NumPy) | Cosine similarity (automatic) |
| **Top-K Control** | Manual sort + slice | `similarity_top_k` parameter |
| **Score Visibility** | ✅ Explicit in output | ✅ `result.score` attribute |
| **Lines of Code** | ~25 lines (`retrieval_utils.py`) | 1 line |

**Observed Scores (Vault Experiment):**
- `"What is the secret password?"` → Top result: `0.7891` ✅ Correct fact returned first.
- `"What is inside the vault?"` (top_k=2) → Missed `"42 golden bars"` ❌ (ranked 3rd).
- `"What is inside the vault?"` (top_k=3) → **All facts retrieved correctly** ✅.

**Key Insight:** `similarity_top_k` is the most important tuning knob in any RAG system. Too low → missing context. Too high → noise and hallucination risk.

---

## 4. Response Generation (Synthesis)

| Dimension | Pure RAG (Manual) | LlamaIndex (`QueryEngine`) |
|:---|:---|:---|
| **Prompt Engineering** | Manual — we wrote the template | Automatic (`CompactAndRefine`) |
| **Multi-fact Synthesis** | Hard — requires careful template design | Automatic — combines all `top_k` nodes |
| **Source Attribution** | Manual | `response.source_nodes` built-in |
| **LLM Used** | `models/gemini-2.0-flash` | `models/gemini-2.0-flash` |

**Key Insight:** In Pure RAG, combining two separate facts into one coherent answer required careful prompt engineering. LlamaIndex's `CompactAndRefine` synthesizer does this automatically.

---

## 5. Architecture Decision Guide

```
Should you use Pure RAG or LlamaIndex?

Is this a learning exercise to understand RAG mechanics?
    └── YES → Pure RAG. Build it from scratch. Suffer. Learn.

Is this a production system or a fast prototype?
    └── YES → LlamaIndex. Configure don't code.

Do you need total control over every retrieval step?
    └── YES → Pure RAG (or LlamaIndex with custom components).

Do you need source attribution and evidence tracking?
    └── LlamaIndex → response.source_nodes is built-in.
```

---

## 6. Lines of Code Comparison

| Pipeline Stage | Pure RAG | LlamaIndex |
|:---|:---:|:---:|
| Chunking | ~40 lines | 3 lines |
| Embedding | ~20 lines | 1 line (`Settings`) |
| Indexing / Storage | ~15 lines | 1 line |
| Retrieval | ~25 lines | 1 line |
| Generation | ~15 lines | 1 line |
| **Total** | **~115 lines** | **~7 lines** |

> **16x less code** — but LlamaIndex is doing the same work underneath. The math never disappeared; the abstraction just hid it.

---

## 7. Final Verdict

| | Pure RAG | LlamaIndex |
|:---|:---:|:---:|
| **Best for learning** | ✅ | ❌ |
| **Best for production** | ❌ | ✅ |
| **Transparency** | ✅ Full | ⚠️ Need to opt-in |
| **Speed to prototype** | ❌ Slow | ✅ Fast |
| **Customizability** | ✅ Total | ✅ High (via components) |

**Bottom Line:** Learn Pure RAG to *understand* RAG. Use LlamaIndex to *build* RAG. They are not competitors — they are different tools for different stages of your journey as an AI Engineer.
