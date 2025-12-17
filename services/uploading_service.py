import os
import uuid
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai.embeddings import  OpenAIEmbeddings
from utils.config import config
from langchain_text_splitters import RecursiveCharacterTextSplitter

def upload_file(file_bytes: bytes, filename: str):
    # 1️⃣ Save file temporarily
    os.makedirs(config.upload_dir, exist_ok=True)
    temp_path = os.path.join(config.upload_dir, f"{uuid.uuid4()}_{filename}")

    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    # 2️⃣ Load PDF
    loader = PyPDFLoader(temp_path)
    docs = loader.load()

    # 3️⃣ Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    # 4️⃣ Store in ChromaDB
    embeddings = OpenAIEmbeddings(model=config.embedding_model_name, 
                                      api_key=config.open_api_key)

    vectorstore = Chroma(collection_name=config.collection_name,
                         embedding_function=embeddings,
                         persist_directory=config.persist_directory)


    vectorstore.add_documents(chunks)
    # return vectorstore

    return {"chunks_added": len(chunks)}
