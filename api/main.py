from fastapi import FastAPI
from api.v1 import query, upload

app = FastAPI(title="Agentic RAG API")

app.include_router(upload.router)
app.include_router(query.router)

@app.get("/health")
def health():
    return {"status": "ok"}
