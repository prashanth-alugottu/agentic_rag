# from backend.src.core.logging_config import setup_logging
from backend.logger.custom_logger import CustomLogger
logger = CustomLogger().get_logger(__file__)

def main():
    print("Hello from agentic-rag!")
    logger.info("App started Al the best ")
    logger.info("Hello Again!")


if __name__ == "__main__":
    main()
