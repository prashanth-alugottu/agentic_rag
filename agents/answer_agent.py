from langchain_openai import ChatOpenAI

def get_answer_agent():
    """
    This agent is responsible ONLY for generating answers
    using the LLM (OpenAI).
    It does NOT know about vector DB or retrieval.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2
    )
    return llm
