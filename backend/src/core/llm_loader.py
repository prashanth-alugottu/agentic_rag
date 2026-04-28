from langchain_openai import ChatOpenAI
from backend.utils.config_loader import load_config
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import dspy

config=load_config()
llm_config=config["llm"]["openai"]
def llm_load():
    """
    This methos loads the llm
    """   
    llm = ChatOpenAI(model=llm_config.get("model_name"))
    return llm

def embedding_loader():
    """
    This method will return embeddings model
    """
    embeddings = OpenAIEmbeddings(model=llm_config.get("embeddings"))
    return embeddings

def configure_dsp():
    if dspy.settings.lm is None:
        lm = dspy.LM("openai/gpt-4o-mini")
        dspy.settings.configure(lm=lm)