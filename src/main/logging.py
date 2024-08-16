import logging
import logging.handlers
from pathlib import Path


def logger_thread(q):
    for record in iter(q.get, "STOP"):
        logger = logging.getLogger(record.name)
        logger.handle(record)


def create_logging_config_dict(logfile: Path):
    return {
        "version": 1,
        "formatters": {
            "detailed": {
                "class": "logging.Formatter",
                "format": "%(asctime)s %(levelname)-8s %(processName)-10s %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "ERROR",
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": logfile,
                "mode": "w",
                "formatter": "detailed",
            },
        },
        "loggers": {"log": {"handlers": ["file"]}},
        "root": {"level": "ERROR", "handlers": ["console"]},
    }
