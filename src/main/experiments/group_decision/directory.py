from ....utils import dirname, filename_csv, filename_json
from ...csv_file import CSVFile
from ...directory import Directory
from ..elicitation.config import MIPConfig
from ..elicitation.fieldnames import ConfigFieldnames
from .fieldnames import CollectiveFieldnames, HyperparametersFieldnames, PathFieldnames
from .hyperparameters import GenHyperparameters


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
            R=self.dirs["root"] / "R",
            P=self.dirs["root"] / "P",
        )
        self.csv_files.update(
            collective=CSVFile(
                self.dirs["root"] / "collective_results.csv", CollectiveFieldnames
            ),
            path=CSVFile(self.dirs["root"] / "path_results.csv", PathFieldnames),
            configs=CSVFile(self.dirs["root"] / "configs.csv", ConfigFieldnames),
            hyperparameters=CSVFile(
                self.dirs["root"] / "hyperparameters.csv", HyperparametersFieldnames
            ),
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
        gen: GenHyperparameters,
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
        gen: GenHyperparameters,
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
        gen: GenHyperparameters,
        Mi_id: int,
        n: int,
        same_alt: bool,
        D_id: int,
        config: MIPConfig,
        id: int,
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
        gen: GenHyperparameters,
        Mi_id: int,
        n: int,
        same_alt: bool,
        D_id: int,
        config: MIPConfig,
        id: int,
        it: int,
    ):
        return self.dirs["C"] / filename_csv(locals())

    def R_dir(
        self,
        m: int,
        ntr: int,
        Atr_id: int,
        k: int,
        Mo_id: int,
        group_size: int,
        gen: GenHyperparameters,
        dm_id: int,
        Mi_id: int,
        n: int,
        same_alt: bool,
        D_id: int,
        config: MIPConfig,
        id: int,
    ):
        return self.dirs["R"] / dirname(locals())

    def R_file(
        self,
        m: int,
        ntr: int,
        Atr_id: int,
        k: int,
        Mo_id: int,
        group_size: int,
        gen: GenHyperparameters,
        dm_id: int,
        Mi_id: int,
        n: int,
        same_alt: bool,
        D_id: int,
        config: MIPConfig,
        id: int,
        it: int,
    ):
        return (
            self.R_dir(**{k: v for k, v in locals().items() if k not in ("self", "it")})
            / f"{it}.csv"
        )

    def P(
        self,
        m: int,
        ntr: int,
        Atr_id: int,
        k: int,
        Mo_id: int,
        group_size: int,
        gen: GenHyperparameters,
        dm_id: int,
        Mi_id: int,
        n: int,
        same_alt: bool,
        D_id: int,
        config: MIPConfig,
        Mc_id: int,
        it: int,
        t: int,
    ):
        return self.dirs["P"] / filename_csv(locals())
