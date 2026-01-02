from langchain_openai import ChatOpenAI
from tools import retriever_tool
import db.vector_store as vector_store
from sentence_transformers import CrossEncoder
from graph.state import RAGState

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


CONFIDENCE_THRESHOLD = 0.5

def retrieve_node(state):
    query = state.get("rewritten_query") or state["query"]
    print("\n🧲 Retrieval Agent received query:", query)
    db = vector_store.getChromaDB()
    docs = db.similarity_search(query, k=20)
    print(f"🧲 Retrieved {len(docs)} documents.")
    for doc in docs:
        print("----- Document chunk:", doc.page_content[:100].replace("\n"," "), "...")
    return {"retrieved_docs": docs}


reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

def rerank_node(state: RAGState):
    docs = state["retrieved_docs"]
    query = state["query"]

    if not docs:
        return {"reranked_docs": []}

    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # keep only strong evidence
    top_docs = [
        doc for doc, score in ranked[:5]
        if score > 0.2
    ]

    print(f"🔎 Reranked to {len(top_docs)} top documents.")
    for doc in top_docs:
        print("----- Top Document chunk:", doc.page_content[:100].replace("\n"," "), "...")

    return {"reranked_docs": top_docs}

def generate_node(state: RAGState):
    docs = state["reranked_docs"]

    if not docs:
        return {
            "answer": "Not found in the provided documents.",
            "grounded": False
        }

    context = "\n\n".join(
        [d.page_content for d in docs]
    )

    prompt = f"""
        You are a factual assistant.

        Answer ONLY using the context below.
        If the answer is not present, say:
        "Not found in the provided documents."

        Context:
        {context}

        Question:
        {state['query']}
        """

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "grounded": True
    }
    
def faithfulness_node(state: RAGState):
    if not state["grounded"]:
        return state

    prompt = f"""
            Check if the answer is fully supported by the context.

            Context:
            {state['reranked_docs']}

            Answer:
            {state['answer']}

            Respond only YES or NO.
            """
    verdict = llm.invoke(prompt).content.strip()

    if verdict == "NO":
        return {
            "answer": "Not found in the provided documents.",
            "grounded": False
        }

    return state
