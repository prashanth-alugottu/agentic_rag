from langchain_openai import ChatOpenAI
from backend.utils.config_loader import load_config

def llm_load():
    """
    This methos loads the llm
    """
    config=load_config()
    llm_config=config["llm"]["openai"]
    llm = ChatOpenAI(model=llm_config.get("model_name"))
    return llm