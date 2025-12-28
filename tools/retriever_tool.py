from langchain.tools import tool
import db.vector_store as vector_store

@tool(return_direct=True)
def retrieve_ans(question: str) -> dict:
    """
    Retrieve top-k documents + scores for better answer generation.
    """
    db = vector_store.getChromaDB()
    results = db.similarity_search_with_relevance_scores(question, k=3)  # top3

    if not results:
        return {"contexts": [], "scores": []}

    docs = []
    scores = []

    for doc, score in results:
        docs.append(doc.page_content)
        scores.append(float(score))

    print("\n🛠 Retriever returned", len(docs))
    print("=====>> Docs:", docs, "\n")

    return {
        "contexts": docs,
        "scores": scores
    }
