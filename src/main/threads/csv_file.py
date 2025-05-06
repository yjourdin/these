import csv
from typing import cast

from ...constants import SENTINEL
from ..csv_file import CSVFields, CSVFile


def csv_file_thread(file: CSVFile[CSVFields]):
    with file.path.open("a", newline="") as f:
        writer = csv.DictWriter(f, file.fieldnames, dialect="unix")

        for result in iter(file.queue.get, SENTINEL):
            result = cast(dict[str, str], result)
            writer.writerow(result)
            f.flush()
