from langchain.tools import tool
import db.vector_store as vector_store

@tool(return_direct=True)
def retrieve_ans(question: str) -> dict:
    """
    Retrieve top-k documents + scores for better answer generation.
    """
    print("\n🔎 Retriever Tool received question:", question)
    db = vector_store.getChromaDB()
    docs = db.similarity_search(question, k=20)  # top3
    return {"retrieved_docs": docs}
