from langchain.tools import tool
import utils.Config as config
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
import db.vector_store as vector_store

@tool
def retrieve_ans(question : str):
    """
    Retrieve relevant documents from the Chroma vector store based on the input question.
    """
    db = vector_store.getChromaDB()
    
    docs = db.similarity_search(question,k= 2)
    if not docs:
        return "No relevant documents found."
    else:
        print("Retrieved Documents: ", docs)
        return docs
    
    
    
    