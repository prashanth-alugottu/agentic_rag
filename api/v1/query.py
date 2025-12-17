from fastapi import APIRouter
from pydantic import BaseModel

from services.vector_store import get_retriever
from tools.retriever_tool import build_retriever_tool
from agents.answer_agent import get_answer_agent
from agents.supervisor import get_supervisor
from graph.supervisor_graph import build_supervisor_graph

router = APIRouter(prefix="/api/v1")

class QueryRequest(BaseModel):
    question: str

@router.get("/query")
def query(req: QueryRequest):
    retriever = get_retriever()
    retriever_tool = build_retriever_tool(retriever)

    supervisor = get_supervisor([retriever_tool])
    answer_agent = get_answer_agent()

    graph = build_supervisor_graph(
        supervisor,
        retriever_tool,
        answer_agent
    )

    result = graph.invoke({"question": req.question})

    return {
        "question": req.question,
        "answer": result["answer"]
    }
