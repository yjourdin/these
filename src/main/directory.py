import csv
from pathlib import Path

from .csv_file import CSVFile

RESULTS_DIR = "results"


class Directory:
    def __init__(self, dir: str, name: str):
        self.dirs: dict[str, Path] = {"root": Path(dir, name)}
        self.csv_files: dict[str, CSVFile] = {}

        self.args = self.dirs["root"] / "args.json"
        self.log = self.dirs["root"] / "log.log"
        self.error = self.dirs["root"] / "error.log"
        self.run = self.dirs["root"] / "run.txt"

    def mkdir(self):
        for dir in self.dirs.values():
            dir.mkdir()

        for file in self.csv_files.values():
            with file.path.open("w", newline="") as f:
                writer = csv.DictWriter(f, file.fieldnames, dialect="unix")
                writer.writeheader()
