
from fastapi import FastAPI
from contextlib import asynccontextmanager
from backend.src.db.vector_db import load_vector_db
from backend.src.core.llm_loader import llm_load
from backend.src.graph.workflow import create_graph
from pydantic import BaseModel
from backend.logger.custom_logger import CustomLogger
import time

log = CustomLogger().get_logger(__file__)

class QueryRequest(BaseModel):
    query:str
    ground_truth: str


@asynccontextmanager
async def lifespan(app: FastAPI):

    vector_db, bm25 = load_vector_db()
    graph = create_graph(bm25, vector_db)
    app.state.graph = graph
    log.info("✅ Graph ready")
    yield

app = FastAPI(lifespan=lifespan)
# app = FastAPI()

@app.get("/")
def root():

    return {"status": "ok"}

@app.post("/query")
def user_query(req:QueryRequest):
    start = time.perf_counter()
    query=req.query
    ground_truth= req.ground_truth
    log.info(f"Query from the user. : {query}")
    graph = app.state.graph
    result = graph.invoke({
        "query": query,
        "ground_truth":ground_truth
    })

    answer = result["answer"]
    log.info(f"Final generated ans : {answer}")
    # return answer
    end = time.perf_counter()
    latency = end - start

    return {
        "query": req.query,
        "answer": result["answer"],
        "latency_seconds": round(latency, 3)
    }
        