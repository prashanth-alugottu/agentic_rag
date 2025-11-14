import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

class Config:
    open_api_key: str = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

    embedding_model_name: str = os.getenv("EMBEDDING_MODEL") or st.secrets.get("EMBEDDING_MODEL")
    collection_name: str = os.getenv("COLLECTION_NAME") or st.secrets.get("COLLECTION_NAME")
    persist_directory: str = os.getenv("PERSIST_DIR") or st.secrets.get("PERSIST_DIR")
    chat_model: str = os.getenv("OPENAI_MODEL") or st.secrets.get("OPENAI_MODEL")

config = Config()
