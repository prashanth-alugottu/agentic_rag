
from typing import TypedDict, List
from langchain_core.documents import Document


class RAGState(TypedDict):
    query: str
    rewritten_query: str
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    answer: str
    grounded: bool