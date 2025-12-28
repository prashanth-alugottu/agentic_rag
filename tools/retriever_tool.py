# tools/retriever_tool.py
from langchain.tools import tool
import db.vector_store as vector_store

@tool(return_direct=True)
def retrieve_ans(question: str) -> dict:
    """
    Retrieves top document with cosine similarity score.
    Returns cleaned result for agent use.
    """
    db = vector_store.getChromaDB()
    results = db.similarity_search_with_relevance_scores(question, k=3)

    if not results:
        return {"context": "", "score": 0}

    # results → [(Document, score), (Document, score)...]
    best_doc, best_score = sorted(results, key=lambda x: x[1], reverse=True)[0]
    print(f"\n🛠 Retriever Tool - Best Doc Score: {best_score}\n")

    return {
        "context": best_doc.page_content,
        "score": best_score
    }
