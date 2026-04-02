import logging
from pathlib import Path

from helpers.config import LOGS_DIR


def setup_logger(verbose: bool = True) -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    log_path = Path(LOGS_DIR) / "app.log"

    logger = logging.getLogger("ai_lab")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    if logger.handlers:
        return logger

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger
