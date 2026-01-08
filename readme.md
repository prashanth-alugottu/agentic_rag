# 🧠 RAG-Based Question Answering System

This project implements a **simple, production-style RAG (Retrieval-Augmented Generation)** system with a focus on **better context selection and reduced hallucination**.

---

## 🔥 Cross-Encoder Reranking (Main Technique Used)

### How it works

- Query and document chunk are passed **together**
- Model applies **full attention**
- Produces a **relevance score**

### Why it is important

- Understands deep meaning between query and text
- Works well for long and complex questions
- Selects the **most relevant context** for the LLM

### Pros

- Very high accuracy
- Strong hallucination reduction
- Best for complex queries

### Cons

- Slower than embedding-based methods
- Costly at large scale

## ⚡ Other Reranking Approaches (Brief)

### Bi-Encoder Reranking

- Encodes query and documents separately
- Faster and cheaper
- Slightly less accurate

Best for:

- Large datasets
- Low-latency systems

---

### LLM-Based Reranking

- LLM ranks passages using reasoning
- Very intelligent but expensive

Used in:

- Legal
- Medical
- High-risk applications

---

### Hybrid Reranking

- Combines BM25 + Vector similarity + Cross-Encoder
- Used in search engines and e-commerce

---

## 🛑 Pre-Rerank Filtering

Before reranking:

- Remove very small chunks
- Remove headers and boilerplate text
- Remove irrelevant content

Benefit:

- Better reranking quality
- Less noise and compute cost
