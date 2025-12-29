from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from agents.agent import retrieval_agent_fn, generator_agent_fn

class RagState(TypedDict):
    query: str
    contexts: list[str]
    scores: list[float]
    answer: str

def build_multirag_graph():
    # 1️⃣ Create graph with schema
    graph = StateGraph(RagState)

    # 2️⃣ Add nodes
    graph.add_node("retrieve", retrieval_agent_fn)
    graph.add_node("generate", generator_agent_fn)

    # 3️⃣ Define flow
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    # 4️⃣ COMPILE (THIS IS CRITICAL)
    return graph.compile()
