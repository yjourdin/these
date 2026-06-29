from pathlib import Path
from typing import Literal

from src.methods import MethodEnum
from src.models import GroupModelEnum
from src.utils import filename_csv, filename_json

from ...csv_file import CSVFile
from ...directory import Directory, DirectoryCSVFiles, DirectoryDirs
from .config import Config
from .csv_files import ConfigCSVFile, TestCSVFile, TrainCSVFile

DirectoryElicitationDirs = DirectoryDirs | Literal["A_train", "A_test", "Mo", "D", "Me"]
DirectoryElicitationCSVFiles = DirectoryCSVFiles | Literal["train", "test", "configs"]


class DirectoryElicitation(Directory):
    def __init__(self, name: str, dir: Path | None = None):
        self.dirs: dict[DirectoryElicitationDirs, Path]
        self.csv_files: dict[DirectoryElicitationCSVFiles, CSVFile]
        super().__init__(name, dir)

        self.dirs |= {  # pyright: ignore[reportIncompatibleVariableOverride]
            "A_train": self.dirs["root"] / "A_train",
            "A_test": self.dirs["root"] / "A_test",
            "Mo": self.dirs["root"] / "Mo",
            "D": self.dirs["root"] / "D",
            "Me": self.dirs["root"] / "Me",
        }

        self.seeds = self.dirs["root"] / "seeds.json"

        self.csv_files = {  # pyright: ignore[reportIncompatibleVariableOverride]
            "train": TrainCSVFile(self.dirs["root"] / "train_results.csv"),
            "test": TestCSVFile(self.dirs["root"] / "test_results.csv"),
            "configs": ConfigCSVFile(self.dirs["root"] / "configs.csv"),
        }

    def A_train(self, m: int, n: int, id: int):
        return self.dirs["A_train"] / filename_csv(locals())

    def A_test(self, m: int, n: int, id: int):
        return self.dirs["A_test"] / filename_csv(locals())

    def Mo(self, m: int, model: GroupModelEnum, k: int, group_size: int, id: int):
        return self.dirs["Mo"] / filename_json(locals())

    def D(
        self,
        m: int,
        ntr: int,
        Atr_id: int,
        Mo: GroupModelEnum,
        k: int,
        group_size: int,
        Mo_id: int,
        n: int,
        same_alt: bool,
        e: float,
        dm_id: int,
        id: int,
    ):
        return self.dirs["D"] / filename_csv(locals())

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
        return self.dirs["Me"] / filename_json(locals())
