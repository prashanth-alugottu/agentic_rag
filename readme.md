

Steps to run the project

1. Open the Project
2. run "uv sync" in comand prompt
3. mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5001

  Here port number should be same as in the code in ml_logs.py file
    mlflow.set_tracking_uri("http://127.0.0.1:5001")

4. uvicorn backend.src.api.app:app --reload 

=======================================================================================
We are goint to use the following:

1. MCP api hit
2. Logging - completed                  -- Completed
3. Readme strp by step                   -- Completed
4. pyproject.toml instead
 of requirements.txt - completed          -- Completed
5. using uv instead of pip                -- Completed
6. Docker deployement
7. CI/CD
8. Langgraph                               -- Completed
9. config/
    settings.yaml                          -- Completed

10. RAG evaluation script.
 --> scripts/evaluate_rag.py
    faithfulness
    context relevance
    answer correctness                       -- Completed

11. LangSmith                                -- Completed
12. OpenAI api and Embeddings                 -- Completed
13. Fast API                             -- Completed
14. Custome Exceptions                       -- Completed
15. Cacahing, Manitoring
16. Cost for each query and disply some where
17. Memory
18. Hybdrid search                              -- Completed
19. Retrival --> Rerranking we 
can use miniLM or SLM(1b param)                     -- Completed
20. Local vector store using FAISS                  -- Completed
21. Kafka
22. SAGE MAKER --> Very Veru Imp