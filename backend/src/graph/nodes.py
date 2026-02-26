from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from backend.src.graph.state import RAGState
from backend.logger.custom_logger import CustomLogger
from backend.exception.custom_exception import AppException
from sentence_transformers import CrossEncoder
from backend.src.core.llm_loader import llm_load

log = CustomLogger().get_logger(__file__)

def bm25_node(bm25):
    def _node(state):
        docs = bm25.invoke(state["query"])
        return {"sparse_docs": docs}
    return _node


def vector_node(vector_db):
    def _node(state):
        docs = vector_db.similarity_search(state["query"], k=10)
        return {"dense_docs": docs}
    return _node


def merge_node(state):
    seen = set()
    merged = []
    for d in state["sparse_docs"] + state["dense_docs"]:
        key = d.page_content
        if key not in seen:
            seen.add(key)
            merged.append(d)
    return {"merged_docs": merged}


reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_node(state):
    query = state["query"]
    docs = state["merged_docs"]

    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    
    ranked = sorted(zip(scores, docs), reverse=True)
    reranked_docs = [d for _, d in ranked]
    
    return {"reranked_docs": reranked_docs}

def topk_node(state):
    return {"context": state["reranked_docs"][:5]}

def generate_node(state):
    context_text = "\n\n".join(d.page_content for d in state["context"])


    prompt = f"""
        You are an expert AI assistant answering questions from retrieved documents.

        Instructions:
        - Answer ONLY from the provided context.
        - Do not infer or assume information not explicitly stated.
        - If the context does not contain the answer, respond:
        "I don't know based on the provided context."
        - Keep the answer clear and concise.
        - If possible, quote or reference relevant phrases from context.

        Context:
        {context_text}

        User Question:
        {state['query']}

        Final Answer:
        """
    log.info(f"Final Prompt is : {prompt}")
    llm=llm_load()
    resp = llm.invoke(prompt)
    return {"answer": resp.content}