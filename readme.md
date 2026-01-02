# 🧠 Production-Style RAG System

This project implements a **production-grade Retrieval-Augmented Generation (RAG)** pipeline with strong focus on **answer correctness, hallucination reduction, and modular design**.

---

## 🏗️ Tech Stack & Core Components

- **Retrieval-Augmented Generation (RAG)**
- **:contentReference[oaicite:0]{index=0}** – Agentic workflow orchestration
- **Cross-Encoder Reranking** – High-precision context selection
- **:contentReference[oaicite:1]{index=1}** – Interactive UI & rapid prototyping

---

## 🔥 1. Cross-Encoder Reranking (Used in This Project)

### 🔍 How it Works

- Query and document chunk are passed **together**
- Model applies **full cross-attention**
- Outputs a **direct relevance score**

### 💡 Why It’s Powerful

- Understands deep semantic relationships
- Handles multi-sentence and multi-hop reasoning
- Selects the **most contextually correct evidence**

### ✅ Pros

- Highest relevance accuracy
- Strong hallucination reduction
- Excellent for complex or ambiguous questions

### ❌ Cons

- Slower inference
- Costly at large scale

### 🏢 Used By

- ChatGPT-style RAG systems
- Perplexity-like search engines
- Enterprise knowledge assistants

> 📌 **Gold standard for reranking when answer quality matters most**

---

## ⚡ 2. Bi-Encoder Reranking (Lightweight Alternative)

### 🔍 How it Works

- Encode query and documents **separately**
- Compute similarity using **vector distance**

### ⚙️ Characteristics

- Faster than cross-encoders
- No token-level interaction → less expressive

### ⚖️ Trade-Off

- Better latency & cost
- Slight drop in relevance accuracy

> 📌 **Best for large-scale systems where speed > precision**

---

## 🧠 3. LLM-Based Reranking (Reasoning-Driven)

### 🧪 Example Prompt
