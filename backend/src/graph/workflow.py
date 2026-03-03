from langgraph.graph import StateGraph, END
from backend.src.graph.state import RAGState
from backend.src.graph.nodes import bm25_node
from backend.src.graph.nodes import vector_node
from backend.src.graph.nodes import merge_node
from backend.src.graph.nodes import rerank_node
from backend.src.graph.nodes import topk_node
from backend.src.graph.nodes import generate_node
from backend.src.graph.nodes import evaluate_rag

def create_graph(bm25, vector_db):
    """"Construct and compiles the langgraph workflow
        Returns:
        Compile Graph: runable graph object for execution
    """
    # initiallize the graph with state schema
    workflow=StateGraph(RAGState)

    # Adding nodes into graph
    workflow.add_node("bm25",bm25_node(bm25))
    workflow.add_node("vector_node",vector_node(vector_db))
    workflow.add_node("merge_node",merge_node)
    workflow.add_node("rerank_node",rerank_node)
    workflow.add_node("topk_node",topk_node)
    workflow.add_node("generate_node",generate_node)
    workflow.add_node("evaluation",evaluate_rag)


    # define the entry point
    workflow.set_entry_point("bm25")

    workflow.add_edge("bm25","vector_node")
    workflow.add_edge("vector_node","merge_node")
    workflow.add_edge("merge_node","rerank_node")
    workflow.add_edge("rerank_node","topk_node")
    workflow.add_edge("topk_node","generate_node")
    workflow.add_edge("generate_node","evaluation")
    workflow.add_edge("evaluation",END)

    # compiling
    app=workflow.compile()
    return app

