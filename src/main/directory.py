import csv
from pathlib import Path
from typing import Literal

from .csv_file import CSVFile
from .csv_files import TaskCSVFile

RESULTS_DIR = Path("results")

DirectoryDirs = Literal["root"]
DirectoryCSVFiles = Literal["tasks"]


class Directory:
    def __init__(self, name: str, dir: Path | None = None):
        self.dirs: dict[DirectoryDirs, Path] = {"root": (dir or Path.cwd()) / name}

        self.args = self.dirs["root"] / "args.json"
        self.log = self.dirs["root"] / "log.log"
        self.error = self.dirs["root"] / "error.log"
        self.run = self.dirs["root"] / "run.txt"

        self.csv_files: dict[DirectoryCSVFiles, CSVFile] = {
            "tasks": TaskCSVFile(self.dirs["root"] / "tasks.csv")
        }

    def iterdir(self):
        return self.dirs.values()

    def itercsv(self):
        return self.csv_files.values()

    def mkdir(self):
        for dir in self.iterdir():
            dir.mkdir()

        for file in self.itercsv():
            with file.path.open("w", newline="") as f:
                writer = csv.DictWriter(f, file.fieldnames, dialect="unix")
                writer.writeheader()

    def close(self):
        for csv_file in self.itercsv():
            csv_file.close()

        self.run.unlink(True)
