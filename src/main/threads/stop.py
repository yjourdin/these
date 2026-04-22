from pathlib import Path

from ..connection import StopEvent


def stopping_thread(stop_event: StopEvent, file: Path):
    while file.exists():
        stop_event.wait(1)
    file.unlink(True)
    stop_event.set()
