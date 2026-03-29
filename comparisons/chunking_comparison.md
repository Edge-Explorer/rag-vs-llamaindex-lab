# Chunking Strategy Comparison: Pure RAG vs LlamaIndex

A deep dive into how each approach splits text into retrievable units.

---

## 1. Overview

Chunking is the **most critical** and **most overlooked** step in a RAG pipeline. The quality of your chunks directly determines the quality of your retrieval — and therefore the quality of your final answer.

> "Garbage in, garbage out. Bad chunks = bad answers."

---

## 2. Approach 1: Pure RAG — Manual Sliding Window

**File:** `pure_rag/chunking_utils.py`

### How it works:
```
Original Text: "Retrieval-Augmented Generation is a powerful technique..."
                 ↓
Window 1 → chars[0:50]
Window 2 → chars[30:80]   ← slides back 20 chars (overlap)
Window 3 → chars[60:110]
...
```

### Implementation:
```python
def sliding_window_chunk(text, chunk_size=50, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - chunk_overlap)
    return chunks
```

### Key Properties:
| Property | Value |
|:---|:---|
| **Unit** | Characters |
| **Boundary Awareness** | ❌ None — cuts mid-word if needed |
| **Overlap** | Configurable (default: 20 chars) |
| **Predictability** | ✅ Very — exact character counts |
| **Risk** | Can split `"Generation"` into `"Generat"` + `"ion"` |

---

## 3. Approach 2: LlamaIndex — `SentenceSplitter`

**File:** `llamaindex_rag/02_custom_chunking.ipynb`

### How it works:
```
Original Text → Tokenized → Grouped by ~50 tokens → Split at sentence boundaries
                 ↓
Node 1: "RAG (Retrieval-Augmented Generation) is a powerful technique for AI.
         RAG (Retrieval-Augmented Generation) is a powerful technique for AI."
Node 2:  ...overlaps by 10 tokens from Node 1...
```

### Implementation:
```python
splitter = SentenceSplitter(chunk_size=50, chunk_overlap=10)
nodes = splitter.get_nodes_from_documents([doc])
```

### Key Properties:
| Property | Value |
|:---|:---|
| **Unit** | Tokens (~4 chars each) |
| **Boundary Awareness** | ✅ Respects `.`, `!`, `\n` sentence endings |
| **Overlap** | Configurable in tokens |
| **Predictability** | High — but can vary slightly by sentence length |
| **Risk** | Very low — only splits mid-sentence if sentence exceeds chunk_size |

---

## 4. Experiment Results (02_custom_chunking.ipynb)

**Input:** 20 repetitions of `"RAG (Retrieval-Augmented Generation) is a powerful technique for AI. "` (≈1,400 characters)

**Settings:** `chunk_size=50`, `chunk_overlap=10`

| Metric | Pure RAG (Character) | LlamaIndex (Token) |
|:---|:---:|:---:|
| **Chunks produced** | ~28 | **10** |
| **Avg chunk size** | 50 chars | ~140 chars |
| **Word boundaries respected** | ❌ Not guaranteed | ✅ Yes |
| **Sentence boundaries respected** | ❌ No | ✅ Yes |

**Why did LlamaIndex produce fewer chunks?**
Because `chunk_size=50` means **50 tokens**, and 1 token ≈ 4 characters. So LlamaIndex's window is actually `50 × 4 = ~200 characters`, which is much larger than our manual 50-character window.

---

## 5. Impact on Retrieval Quality

| Scenario | Pure RAG | LlamaIndex |
|:---|:---|:---|
| Short, precise sentences | ✅ Works well | ✅ Works well |
| Long technical paragraphs | ⚠️ May cut mid-concept | ✅ Handles better |
| Code snippets | ⚠️ No boundary awareness | ⚠️ May not detect code blocks |
| Multi-language text | ⚠️ Character count varies | ✅ Token-aware handles better |

---

## 6. Key Learnings

1. **Token ≠ Character**: LlamaIndex's `chunk_size=50` is ~4× larger than Pure RAG's `chunk_size=50` in actual text volume. Always clarify your unit.

2. **Fewer, cleaner chunks beat many scrappy chunks.** LlamaIndex's 10 semantically complete nodes outperform 28 mid-word fragments.

3. **`chunk_overlap` is the "memory glue"** — without it, a concept that spans two chunks is invisible to the retriever.

4. **Always inspect your nodes.** Use `splitter.get_nodes_from_documents()` to see exactly what the vector database will receive before you build the index.

---

## 7. Recommendations

| Use Case | Recommended Strategy |
|:---|:---|
| Learning / Experiments | Pure RAG character chunker |
| Production APIs | LlamaIndex `SentenceSplitter` |
| Code Documentation | `CodeSplitter` (LlamaIndex) |
| Markdown / Structured Docs | `MarkdownNodeParser` (LlamaIndex) |
