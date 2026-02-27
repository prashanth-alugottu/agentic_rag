from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from backend.utils.config_loader import load_config
import os
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()
def load_vector_db():
    """ 
    This method will retun the FAISS db instance with loading the local db
    """ 
    config=load_config()
    llm_config=config["llm"]["openai"]
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "mps"}
    )

    vector_db = FAISS.load_local(
        llm_config.get("vector_db"),
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Build BM25 from same docs
    bm25 = build_bm25_from_docs(vector_db)
    return vector_db,bm25


def build_bm25_from_docs(vector_db):
    """
    Build BM25 retriever from documents stored in FAISS vector store.
    
    Args:
        vector_db: FAISS vector store
    
    Returns:
        BM25Retriever
    """
    # Extract stored documents from FAISS docstore
    docs = list(vector_db.docstore._dict.values())
    # store=vector_db.docstore._dict
    # for doc_id, doc in store.items():
    #     if "2,00,00,000" in doc.page_content:
    #         print()
    #         print("FOUND:", doc.page_content)
    #         print()

    # Build BM25
    bm25 = BM25Retriever.from_documents(docs)
    # Optional: number of results to return
    bm25.k = 10
    return bm25