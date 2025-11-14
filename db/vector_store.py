# Updated retriever.py - Using current Cohere embedding model


from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
import os
from langchain_openai.embeddings import  OpenAIEmbeddings
from utils.config import config

def upload_file(file :str):
    """Loads documents, splits them, creates embeddings, and stores in Chroma vector store"""
   
    file_path = f"docs/{file}"
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    chunks = split_text(documents)
    embeddings = OpenAIEmbeddings(model=config.embedding_model_name, 
                                      api_key=config.open_api_key)
    
    vectorstore = Chroma(collection_name=config.collection_name,
                         embedding_function=embeddings,
                         persist_directory=config.persist_directory)
    
    vectorstore.add_documents(chunks)
    return vectorstore


from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text(documents):
    """Splits documents into smaller chunks for embedding"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return text_splitter.split_documents(documents)




def get_vector_store():
    """
    Return a handle to the existing Chroma vector store.
    """
    # Initialize the same embedding model used during DB creation
    embeddings = OpenAIEmbeddings(model=config.embedding_model_name, 
                                      api_key=config.open_api_key)

    # Connect to the existing persisted vector store
    vector_store = Chroma(
        collection_name=config.collection_name,
        embedding_function=embeddings,
        persist_directory=config.persist_directory  # Path where DB was saved
    )

    return vector_store



def getChromaDB():
    """Returns the Chroma vector store instance."""
    embeddings = OpenAIEmbeddings(model=config.embedding_model_name, 
                                      api_key=config.open_api_key)
    
    db = Chroma(
        persist_directory=config.persist_directory,
        embedding_function=embeddings,
        collection_name=config.collection_name
    )
    return db