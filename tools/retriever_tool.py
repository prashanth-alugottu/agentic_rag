from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
import services.vector_store as vector_store


from langchain.tools import tool

def build_retriever_tool(retriever):

    @tool
    def retrieve_documents(query: str) -> str:
        """
        Retrieve relevant documents from ChromaDB.
        """
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return ""
        return "\n\n".join([d.page_content for d in docs])

    return retrieve_documents

    
    
    