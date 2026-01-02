from langgraph.graph import StateGraph
from graph.state import RAGState

from agents.agent import retrieve_node
from agents.agent import rerank_node
from agents.agent import generate_node
from agents.agent import faithfulness_node


def build_multirag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)
    graph.add_node("faithfulness", faithfulness_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", "faithfulness")

    return graph.compile()
