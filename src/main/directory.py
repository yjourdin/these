import csv
from pathlib import Path

from ..methods import MethodEnum
from ..models import GroupModelEnum
from ..utils import filename_csv, filename_json
from .config import Config
from .csv_file import CSVFile, CSVFiles
from .fieldnames import (
    ConfigFieldnames,
    DSizeFieldnames,
    SeedFieldnames,
    TestFieldnames,
    TrainFieldnames,
)

RESULTS_DIR = "results"


class Directory:
    def __init__(self, dir: str, name: str):
        self.root_dir = Path(dir, name)
        self.A_train_dir = self.root_dir / "A_train"
        self.A_test_dir = self.root_dir / "A_test"
        self.Mo_dir = self.root_dir / "Mo"
        self.D_dir = self.root_dir / "D"
        self.Me_dir = self.root_dir / "Me"
        self.args = self.root_dir / "args.json"
        self.log = self.root_dir / "log.log"
        self.error = self.root_dir / "error.log"
        self.run = self.root_dir / "run.txt"
        self.csv_files = CSVFiles(
            {
                "train": CSVFile(self.root_dir / "train_results.csv", TrainFieldnames),
                "test": CSVFile(self.root_dir / "test_results.csv", TestFieldnames),
                "seeds": CSVFile(self.root_dir / "seeds.csv", SeedFieldnames),
                "configs": CSVFile(self.root_dir / "configs.csv", ConfigFieldnames),
                "D_size": CSVFile(self.root_dir / "D_size.csv", DSizeFieldnames),
            }
        )

    def A_train(self, m: int, n: int, id: int):
        return self.A_train_dir / filename_csv(locals())

    def A_test(self, m: int, n: int, id: int):
        return self.A_test_dir / filename_csv(locals())

    def Mo(self, m: int, model: GroupModelEnum, k: int, group_size: int, id: int):
        return self.Mo_dir / filename_json(locals())

    def D(
        self,
        m: int,
        ntr: int,
        Atr_id: int,
        Mo: GroupModelEnum,
        ko: int,
        group_size: int,
        Mo_id: int,
        n: int,
        same_alt: bool,
        e: float,
        dm_id: int,
        id: int,
    ):
        return self.D_dir / filename_csv(locals())

    def Me(
        self,
        m: int,
        ntr: int,
        Atr_id: int,
        Mo: GroupModelEnum,
        ko: int,
        group_size: int,
        Mo_id: int,
        n: int,
        same_alt: bool,
        e: float,
        D_id: int,
        Me: GroupModelEnum,
        ke: int,
        method: MethodEnum,
        config: Config,
        id: int,
    ):
        return self.Me_dir / filename_json(locals())

    def mkdir(self):
        self.root_dir.mkdir()
        self.A_train_dir.mkdir()
        self.A_test_dir.mkdir()
        self.Mo_dir.mkdir()
        self.D_dir.mkdir()
        self.Me_dir.mkdir()

        self.run.touch()

        for file in self.csv_files.values():
            with file.path.open("w", newline="") as f:
                writer = csv.DictWriter(f, file.fieldnames, dialect="unix")
                writer.writeheader()
