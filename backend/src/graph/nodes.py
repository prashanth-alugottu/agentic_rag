from backend.logger.custom_logger import CustomLogger
from sentence_transformers import CrossEncoder
from backend.src.core.llm_loader import llm_load
from backend.src.core.llm_loader import configure_dsp
import threading
from backend.src.eval.evaluate import calculate_ragas_metrics
from backend.src.eval.ml_logs import _log_to_mlflow
import dspy
from backend.src.dspy.rag_dspy import RAGSignature, MultiStepRAG, ContextFilter

log = CustomLogger().get_logger(__file__)

def bm25_node(bm25): # 5
    def _node(state): # key word matching  (TF-IDF)
        docs = bm25.invoke(state["query"])
        return {"sparse_docs": docs}
    return _node


def vector_node(vector_db): #5
    def _node(state):
        docs = vector_db.similarity_search(state["query"], k=5)
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

def dspy_rag_node(state):
    query = state["query"]
    docs = state["context"]
    full_context = "\n".join([d.page_content for d in docs])
    
    # Step 2: Filter context (reduce noise)
    context_filter=ContextFilter()
    filtered=context_filter(context=full_context,question=query)
    filtered_context=getattr(filtered, "filtered_context",full_context)
    print("========>>> filtered_context : ", filtered_context)
     # Step 3: Multi-step reasoning
    rag = MultiStepRAG()
    result = rag(context=filtered_context, question=query)

    answer = result.answer
    print("========>>> : ",result)

    # Step 4: Fallback (production safety)
    if not answer or "I don't know" in answer:
        llm = llm_load()
        fallback = llm.invoke(query).content
        return {"answer": fallback}

    return {"answer": answer}

def generate_node(state):
    context_docs = state["context"]
    context_text = "\n\n".join(d.page_content for d in context_docs)


    prompt = f"""
        You are an expert AI assistant answering questions from retrieved documents.

        Instructions:
        - Answer ONLY from the provided context.
        - Do not infer or assume information not explicitly stated.
        - If the context does not contain the answer, respond:
        "I don't know based on the provided context."
        - Keep the answer clear and concise.
        - If possible, quote or reference relevant phrases from context.
        - Keep answer concise and short in 3 to 4 lines max.

        Context:
        {context_text}

        User Question:
        {state['query']}

        Final Answer:
        """
    log.info(f"Final Prompt is : {prompt}")
    llm=llm_load()
    resp = llm.invoke(prompt)
    ans = resp.content
    return {"answer": ans}


def evaluate_rag(state):
    answer = state["answer"]

    if not state.get("ground_truth"):
        return {}

    threading.Thread(target=background_eval,args=(state,answer), daemon=True).start()

    return {}

def background_eval(state,answer):
    try:
        contexts = [doc.page_content for doc in state["context"]]

        result = calculate_ragas_metrics(
            state["query"],
            answer,
            contexts,
            state["ground_truth"],
        )

        log.info(f"RAGAS result: {result}")

        _log_to_mlflow(result, state)

    except Exception as e:
        log.error(f"Background RAGAS error: {e}")