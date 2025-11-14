from langchain.chat_models import init_chat_model
import os
from utils.config import config
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent   
import tools.retriever_tool as retriever_tool

def retrieve_agent(user_query: str):
    """ Create and return a Quiz Generation Agent."""
    model = ChatOpenAI(model=config.chat_model, temperature=0)
    
    systsem_prompt = """You are an intelligent assistant that helps users by 
    retrieving relevant documents from a vector store based on their questions. 
    Use the provided tool to fetch the most pertinent information and provide concise, 
    accurate answers.
    
    - if the user query is empty, respond with "Please provide a valid question."
    - if no relevant documents are found, respond with "I'm sorry, I couldn't find any relevant information in the documents provided."
    - Always use the tool to retrieve documents before answering.
    - if the document. is not yet uploaded , respond with "No documents have been uploaded yet. Please upload a document to proceed."
    
    
    """
    context_schema = "Use the tool to retrieve relevant documents from the vector store based on the input question."
    agent = create_agent(model = model,
                         tools=[retriever_tool.retrieve_ans],
                         context_schema=context_schema,
                         system_prompt=systsem_prompt)
    
    result = agent.invoke(
    {"messages": [{"role": "user", "content": user_query}]})
    print("------------------------------------------------")
    print("Sdsdsdd ", result)
    print("------------------------------------------------")
    return result