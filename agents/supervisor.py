from langchain_openai import ChatOpenAI

def get_supervisor(tools):
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    ).bind_tools(tools)
