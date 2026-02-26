from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.logger.custom_logger import CustomLogger
from backend.exception.custom_exception import AppException
from langchain_community.embeddings import HuggingFaceEmbeddings
from backend.utils.config_loader import load_config
import os
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

log = CustomLogger().get_logger(__file__)

def loading_data():
    """
    Loading the data from the data folder
    """
    try:
        pdf_path = f"{Path(__file__).resolve().parents[1]}/data/Policy.pdf"
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(documents)
        log.info("Documents are loaded")
        return documents
    except Exception as e:
        log.info("Error occured while loading the data : {e}")
        raise AppException("Error occured while loading the data : {e}")
    
def save_data_local(raw_docs):
    """ Chunking the documents """
    text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1000,
                chunk_overlap = 200
            )
    config=load_config()
    llm_config=config["llm"]["openai"]

    splits = text_splitter.split_documents(raw_docs)
    log.info(f"Chunks are : {splits}")
    log.info(f"=======>>>>. {llm_config.get('embeddings')}")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "mps"}  # Mac GPU
    )
    db = FAISS.from_documents(splits, embeddings)
    db.save_local(llm_config.get("vector_db"))

data=loading_data()
save_data_local(data)