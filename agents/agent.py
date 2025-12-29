from langchain_openai import ChatOpenAI
from tools import retriever_tool

CONFIDENCE_THRESHOLD = 0.5

def retrieval_agent_fn(state):
    query = state["query"]

    tool_output = retriever_tool.retrieve_ans.invoke(
        {"question": query}
    )

    contexts = tool_output["contexts"]
    scores = tool_output["scores"]
    print(f"\n🛠 Retriever Tool Output - Contexts: {contexts} |===========|  Scores: {scores}")

    # 🔴 RAG GUARD
    if not scores or max(scores) < CONFIDENCE_THRESHOLD:
        state["contexts"] = ['Not covered in documents']
        state["scores"] = []
        state["answer"] = ""
        return state

    state["contexts"] = contexts
    state["scores"] = scores
    return state



def generator_agent_fn(state):
    contexts = state["contexts"]
    query = state["query"]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"Answer using ONLY this context:\n{contexts}\nQuestion: {query}"
    answer = llm.invoke(prompt).content
    state["answer"] = answer
    return state
