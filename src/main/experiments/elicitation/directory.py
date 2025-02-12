from pathlib import Path

from ....methods import MethodEnum
from ....models import GroupModelEnum
from ....utils import filename_csv, filename_json
from ...directory import Directory
from ...experiments.elicitation.config import Config
from .csv_files import (
    ConfigCSVFile,
    TestCSVFile,
    TrainCSVFile,
)


class DirectoryElicitation(Directory):
    class Dirs(Directory.Dirs):
        A_train: Path
        A_test: Path
        Mo: Path
        D: Path
        Me: Path

    class CSVFiles(Directory.CSVFiles):
        train: TrainCSVFile
        test: TestCSVFile
        configs: ConfigCSVFile

    def __init__(self, dir: str, name: str):
        super().__init__(dir, name)

        self.dirs = self.Dirs(
            self.dirs,
            A_train=self.dirs["root"] / "A_train",
            A_test=self.dirs["root"] / "A_test",
            Mo=self.dirs["root"] / "Mo",
            D=self.dirs["root"] / "D",
            Me=self.dirs["root"] / "Me",
        )
        
        self.seeds = self.dirs["root"] / "seeds.json"

        self.csv_files = self.CSVFiles(
            self.csv_files,
            train=TrainCSVFile(self.dirs["root"] / "train_results.csv"),
            test=TestCSVFile(self.dirs["root"] / "test_results.csv"),
            configs=ConfigCSVFile(self.dirs["root"] / "configs.csv"),
        )

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
