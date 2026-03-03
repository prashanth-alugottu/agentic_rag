# from backend.src.core.logging_config import setup_logging
from backend.logger.custom_logger import CustomLogger
from pathlib import Path
import os
from dotenv import load_dotenv
from backend.script.ingestion import loading_data, save_data_local

log = CustomLogger().get_logger(__file__)

def main():
    raw_data=loading_data()
    save_data_local(raw_data)

if __name__ == "__main__":
    main()
