import csv
from pathlib import Path
from typing import Any

from ..homotypeddict import HomoTypedDict
from .csv_file import CSVFile
from .csv_files import TaskCSVFile

RESULTS_DIR = Path("results")


class Directory:
    class Dirs(HomoTypedDict[Path]):
        root: Path

    class CSVFiles(HomoTypedDict[CSVFile[Any]]):
        tasks: TaskCSVFile

    def __init__(self, name: str, dir: Path = Path.cwd()):
        self.dirs = self.Dirs(root=Path(dir, name))

        self.args = self.dirs["root"] / "args.json"
        self.log = self.dirs["root"] / "log.log"
        self.error = self.dirs["root"] / "error.log"
        self.run = self.dirs["root"] / "run.txt"

        self.csv_files = self.CSVFiles(
            tasks=TaskCSVFile(self.dirs["root"] / "tasks.csv"),
        )

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
