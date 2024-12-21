from ....methods import MethodEnum
from ....models import GroupModelEnum
from ....utils import filename_csv, filename_json
from ...csv_file import CSVFile
from ...directory import Directory
from ...experiments.elicitation.config import Config
from .fieldnames import (
    ConfigFieldnames,
    SeedFieldnames,
    TestFieldnames,
    TrainFieldnames,
)


class DirectoryElicitation(Directory):
    def __init__(self, dir: str, name: str):
        super().__init__(dir, name)
        self.dirs.update(
            A_train=self.dirs["root"] / "A_train",
            A_test=self.dirs["root"] / "A_test",
            Mo=self.dirs["root"] / "Mo",
            D=self.dirs["root"] / "D",
            Me=self.dirs["root"] / "Me",
        )
        self.csv_files.update(
            train=CSVFile(self.dirs["root"] / "train_results.csv", TrainFieldnames),
            test=CSVFile(self.dirs["root"] / "test_results.csv", TestFieldnames),
            seeds=CSVFile(self.dirs["root"] / "seeds.csv", SeedFieldnames),
            configs=CSVFile(self.dirs["root"] / "configs.csv", ConfigFieldnames),
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
        ko: int,
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
