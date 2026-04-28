from ragas import evaluate
from backend.src.core.llm_loader import llm_load, embedding_loader
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
    )

from datasets import Dataset

def calculate_ragas_metrics(question,answer,contexts,ground_truth):
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [ground_truth],
    }
    dataset=Dataset.from_dict(data)

    embeddings = embedding_loader()
    llm = llm_load()
    
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        embeddings=embeddings,
        llm=llm,
    )
    return result