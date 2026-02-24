from backend.src.core.logging_config import setup_logging

setup_logging()

import logging
log = logging.getLogger(__name__)
def main():
    print("Hello from agentic-rag!")
    log.info("HI chinnu All the best")


if __name__ == "__main__":
    main()
