import csv
from multiprocessing import Queue
from pathlib import Path

from ...constants import SENTINEL
from ...utils import dict_values_to_str
from ..fieldnames import Fieldnames


def csv_file_thread(file: Path, fieldnames: Fieldnames, q: Queue):
    with file.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames, dialect="unix")
        for result in iter(q.get, SENTINEL):
            writer.writerow(dict_values_to_str(result))
            f.flush()
