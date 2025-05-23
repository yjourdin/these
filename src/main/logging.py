
from multiprocessing import Queue
from typing import Any

from .directory import Directory


type LoggingQueue = "Queue[Any]"


def create_logging_config_dict(dir: Directory):
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
            "logfile": {
                "class": "logging.FileHandler",
                "filename": dir.log,
                "formatter": "detailed",
            },
            "errorfile": {
                "class": "logging.FileHandler",
                "level": "ERROR",
                "filename": dir.error,
                "formatter": "detailed",
            },
        },
        "loggers": {"log": {"handlers": ["logfile", "errorfile"]}},
        "root": {"level": "ERROR", "handlers": ["console"]},
    }
