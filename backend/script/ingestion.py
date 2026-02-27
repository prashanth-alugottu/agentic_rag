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
import fitz
import pdfplumber
from langchain_core.documents import Document
from pathlib import Path

load_dotenv()

log = CustomLogger().get_logger(__file__)



def load_pdf_table_aware(pdf_path: str):
    docs = []
    # -------- TEXT BLOCKS --------
    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks")

        for b in blocks:
            text = b[4].strip()
            if len(text) < 40:
                continue

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": page_num + 1,
                        "type": "text"
                    }
                )
            )

    # -------- TABLES --------
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()

            if not tables:
                continue

            for table in tables:
                if not table:
                    continue

                header = table[0]

                for row_idx, row in enumerate(table):
                    if not row:
                        continue

                    cells = [c.strip() for c in row if c and c.strip()]
                    if len(cells) < 2:
                        continue

                    row_text = " | ".join(cells)

                    docs.append(
                        Document(
                            page_content=row_text,
                            metadata={
                                "page": page_num + 1,
                                "type": "table_row",
                                "row": row_idx
                            }
                        )
                    )

                    # semantic version for embeddings
                    if header and len(header) == len(row):
                        semantic = []
                        for h, r in zip(header, row):
                            if h and r:
                                semantic.append(f"{h.strip()} is {r.strip()}")

                        if semantic:
                            docs.append(
                                Document(
                                    page_content="; ".join(semantic),
                                    metadata={
                                        "page": page_num + 1,
                                        "type": "table_semantic",
                                        "row": row_idx
                                    }
                                )
                            )

    return docs


def loading_data():
    """
    Loading the data from the data folder
    """
    try:
        pdf_path = f"{Path(__file__).resolve().parents[1]}/data/Policy.pdf"
        documents = load_pdf_table_aware(pdf_path)
        # print(documents)
        log.info("Documents are loaded")
        return documents
    except Exception as e:
        log.info("Error occured while loading the data : {e}")
        raise AppException("Error occured while loading the data : {e}")
    
def save_data_local(raw_docs):
    """ Chunking the documents """
    text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 800,
                chunk_overlap = 120
            )
    config=load_config()
    llm_config=config["llm"]["openai"]
    print(f"\n\n\n ====> Raw Docs {raw_docs}")
    splits = text_splitter.split_documents(raw_docs)
    log.info(f"Chunks are : {splits}")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "mps"}  # Mac GPU
    )
    db = FAISS.from_documents(splits, embeddings)
    db.save_local(llm_config.get("vector_db"))

# data=loading_data()
# save_data_local(data)



