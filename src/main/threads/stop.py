from multiprocessing.connection import Connection
from pathlib import Path

from ...constants import SENTINEL


def stopping_thread(file: Path, connection: Connection):
    while file.exists() and not connection.poll():
        pass
    if file.exists():
        file.unlink()
    if connection.poll():
        connection.recv()
    else:
        connection.send(SENTINEL)
