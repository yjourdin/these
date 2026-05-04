import csv
from threading import Thread

from ..csv_file import CSVFields, CSVFile


class CSVFileThread(Thread):
    def __init__(self, file: CSVFile[CSVFields]) -> None:
        self.path = file.path
        self.fieldnames = file.fieldnames
        self.queue = file.queue
        super().__init__(name=str(self.path))
        self.start()

    def run(self):
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, self.fieldnames, dialect="unix")

            for result in iter(self.queue.get, {}):
                writer.writerow(result)
                f.flush()
                self.queue.task_done()

            self.queue.task_done()
