import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

class Config:
    open_api_key: str = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

    embedding_model_name: str = "text-embedding-3-large"
    collection_name: str = "vector_collection"
    persist_directory: str = "./vector_db"
    chat_model: str = "gpt-4o-mini"
    upload_dir: str = "docs/"

config = Config()
