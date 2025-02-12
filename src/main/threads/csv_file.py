import csv
from multiprocessing import Queue
from pathlib import Path

from ...constants import SENTINEL


def csv_file_thread(file: Path, fieldnames: list[str], q: "Queue[dict[str, str]]"):
    with file.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames, dialect="unix")

        for result in iter(q.get, SENTINEL):
            writer.writerow(result)
            f.flush()
