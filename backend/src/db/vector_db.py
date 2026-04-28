from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from backend.utils.config_loader import load_config
import os
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
import pickle

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

    # Load BM25
    with open("db_files/bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

    return vector_db,bm25
