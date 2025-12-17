from langgraph.graph import StateGraph
from typing import TypedDict, Optional

class AgentState(TypedDict):
    question: str
    context: Optional[str]
    answer: Optional[str]

def build_supervisor_graph(supervisor, retriever_tool, answer_agent):

    def supervisor_node(state: AgentState):
        tool_call = supervisor.invoke(state["question"])
        return {"context": tool_call}

    def answer_node(state: AgentState):
        prompt = f"""
                    Answer the question using the context below.
                    If context is empty, say "I don't know".

                    Context:
                    {state.get('context')}

                    Question:
                    {state['question']}
                    """
        result = answer_agent.invoke(prompt)
        return {"answer": result.content}

    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "answer")

    return graph.compile()
