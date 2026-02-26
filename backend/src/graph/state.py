from typing import TypedDict, List
from langchain_core.documents import Document


class RAGState(TypedDict):
    query: str
    sparse_docs: List[Document]
    dense_docs: List[Document]
    merged_docs: List[Document]
    reranked_docs: List[Document]
    context : List[Document]
    answer: str