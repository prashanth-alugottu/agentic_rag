# from backend.src.core.logging_config import setup_logging
from backend.logger.custom_logger import CustomLogger
from backend.utils.config_loader import load_config
logger = CustomLogger().get_logger(__file__)

def main():
    print("Hello from agentic-rag!")
    logger.info("App started Al the best ")
    logger.info("Hello Again!")

    config = load_config()

    # select provider
    llm_cfg = config["llm"]["openai"]
    logger.info(f"LLMs is. : {llm_cfg}")
    provider = {config["llm"]["openai"]["provider"]}
    logger.info(f"LLMs Provider  : {provider}")



if __name__ == "__main__":
    main()
