import mlflow
from backend.logger.custom_logger import CustomLogger

log = CustomLogger().get_logger(__file__)

def _log_to_mlflow(result, state):
    """Runs inside thread executor"""
    try:
        mlflow.set_tracking_uri("http://127.0.0.1:5001")
        mlflow.set_experiment("Hybrid_RAG_Production")
        print(f"Logging streted mlflow : {result}")
         # Extract first element safely
        faithfulness = float(result["faithfulness"][0])
        answer_relevancy = float(result["answer_relevancy"][0])
        context_precision = float(result["context_precision"][0])
        context_recall = float(result["context_recall"][0])
        print(50*"=")
        print(faithfulness,answer_relevancy,context_precision,context_recall)
        with mlflow.start_run(run_name="RAG_Evaluation"):
            mlflow.log_param("question", state["query"])
            mlflow.log_param("ground_truth", state.get("ground_truth", ""))

            mlflow.log_metric("faithfulness", faithfulness)
            mlflow.log_metric("answer_relevancy", answer_relevancy)
            mlflow.log_metric("context_precision", context_precision)
            mlflow.log_metric("context_recall", context_recall)

            # hallucination flag
            hallucinated = 1 if faithfulness < 0.8 else 0
            mlflow.log_metric("hallucinated", hallucinated)
            log.info("END")
    except Exception as e:
        log.error(f"Erro rin log_to_mlfloe : {e}")



