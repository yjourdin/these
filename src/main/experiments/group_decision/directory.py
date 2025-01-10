from ....utils import filename_csv, filename_json
from ...csv_file import CSVFile
from ...directory import Directory
from ..elicitation.config import MIPConfig
from ..elicitation.fieldnames import ConfigFieldnames
from .fieldnames import (
    AcceptFieldnames,
    CleanFieldnames,
    CollectiveFieldnames,
    CompromiseFieldnames,
    GroupParametersFieldnames,
    PathFieldnames,
)
from .fields import GroupParameters


class DirectoryGroupDecision(Directory):
    def __init__(self, dir: str, name: str):
        super().__init__(dir, name)
        self.dirs.update(
            A_train=self.dirs["root"] / "A_train",
            Mo=self.dirs["root"] / "Mo",
            Mi=self.dirs["root"] / "Mi",
            D=self.dirs["root"] / "D",
            Mc=self.dirs["root"] / "Mc",
            C=self.dirs["root"] / "C",
            RP=self.dirs["root"] / "RP",
            RC=self.dirs["root"] / "RC",
            P=self.dirs["root"] / "P",
        )
        self.csv_files.update(
            accept=CSVFile(self.dirs["root"] / "accept_results.csv", AcceptFieldnames),
            clean=CSVFile(self.dirs["root"] / "clean_results.csv", CleanFieldnames),
            collective=CSVFile(
                self.dirs["root"] / "collective_results.csv", CollectiveFieldnames
            ),
            compromise=CSVFile(
                self.dirs["root"] / "compromise_results.csv", CompromiseFieldnames
            ),
            configs=CSVFile(self.dirs["root"] / "configs.csv", ConfigFieldnames),
            group_parameters=CSVFile(
                self.dirs["root"] / "group_parameters.csv", GroupParametersFieldnames
            ),
            path=CSVFile(self.dirs["root"] / "path_results.csv", PathFieldnames),
        )

    def A_train(self, m: int, n: int, id: int):
        return self.dirs["A_train"] / filename_csv(locals())

    def Mo(self, m: int, k: int, id: int):
        return self.dirs["Mo"] / filename_json(locals())

    def Mi(
        self,
        m: int,
        k: int,
        Mo_id: int,
        group_size: int,
        group: GroupParameters,
        dm_id: int,
        id: int,
    ):
        return self.dirs["Mi"] / filename_json(locals())

    def D(
        self,
        m: int,
        ntr: int,
        Atr_id: int,
        k: int,
        Mo_id: int,
        group_size: int,
        group: GroupParameters,
        dm_id: int,
        Mi_id: int,
        n: int,
        same_alt: bool,
        id: int,
        it: int,
    ):
        return self.dirs["D"] / filename_csv(locals())

    def Mc(
        self,
        m: int,
        ntr: int,
        Atr_id: int,
        k: int,
        Mo_id: int,
        group_size: int,
        group: GroupParameters,
        Mi_id: int,
        n: int,
        same_alt: bool,
        D_id: int,
        config: MIPConfig,
        id: int,
        P_id: int,
        it: int,
    ):
        return self.dirs["Mc"] / filename_json(locals())

    def C(
        self,
        m: int,
        ntr: int,
        Atr_id: int,
        k: int,
        Mo_id: int,
        group_size: int,
        group: GroupParameters,
        Mi_id: int,
        n: int,
        same_alt: bool,
        D_id: int,
        config: MIPConfig,
        id: int,
        P_id: int,
        it: int,
    ):
        return self.dirs["C"] / filename_csv(locals())

    def RP(
        self,
        m: int,
        ntr: int,
        Atr_id: int,
        k: int,
        Mo_id: int,
        group_size: int,
        group: GroupParameters,
        dm_id: int,
        Mi_id: int,
        n: int,
        same_alt: bool,
        D_id: int,
        config: MIPConfig,
        id: int,
        P_id: int,
        it: int,
    ):
        return self.dirs["RP"] / filename_csv(locals())

    def RC(
        self,
        m: int,
        ntr: int,
        Atr_id: int,
        k: int,
        Mo_id: int,
        group_size: int,
        group: GroupParameters,
        dm_id: int,
        Mi_id: int,
        n: int,
        same_alt: bool,
        D_id: int,
        config: MIPConfig,
        id: int,
        P_id: int,
        it: int,
    ):
        return self.dirs["RC"] / filename_csv(locals())

    def P(
        self,
        m: int,
        ntr: int,
        Atr_id: int,
        k: int,
        Mo_id: int,
        group_size: int,
        group: GroupParameters,
        dm_id: int,
        Mi_id: int,
        n: int,
        same_alt: bool,
        D_id: int,
        config: MIPConfig,
        Mc_id: int,
        id: int,
        it: int,
        t: int,
    ):
        return self.dirs["P"] / filename_csv(locals())
