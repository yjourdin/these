from pathlib import Path

from ....utils import filename_csv, filename_json
from ...directory import Directory
from ..elicitation.config import MIPConfig
from ..elicitation.csv_files import ConfigCSVFile
from .csv_files import (
    AcceptCSVFile,
    ChangesCSVFile,
    CleanCSVFile,
    CompromiseCSVFile,
    GroupParametersCSVFile,
    MIPCSVFile,
    PathCSVFile,
)
from .fields import GroupParameters


class DirectoryGroupDecision(Directory):
    class Dirs(Directory.Dirs):
        A: Path
        Mo: Path
        Mi: Path
        D: Path
        Di: Path
        Mc: Path
        Dc: Path
        C: Path
        Dr: Path
        Cr: Path
        P: Path

    class CSVFiles(Directory.CSVFiles):
        accept: AcceptCSVFile
        changes: ChangesCSVFile
        clean: CleanCSVFile
        mip: MIPCSVFile
        compromise: CompromiseCSVFile
        configs: ConfigCSVFile
        group_parameters: GroupParametersCSVFile
        path: PathCSVFile

    def __init__(self, name: str, dir: Path = Path.cwd()):
        super().__init__(name, dir)

        self.dirs = self.Dirs(
            self.dirs,
            A=self.dirs["root"] / "A",
            Mo=self.dirs["root"] / "Mo",
            Mi=self.dirs["root"] / "Mi",
            D=self.dirs["root"] / "D",
            Di=self.dirs["root"] / "Di",
            Mc=self.dirs["root"] / "Mc",
            Dc=self.dirs["root"] / "Dc",
            C=self.dirs["root"] / "C",
            Dr=self.dirs["root"] / "Dr",
            Cr=self.dirs["root"] / "Cr",
            P=self.dirs["root"] / "P",
        )

        self.seeds = self.dirs["root"] / "seeds.json"

        self.csv_files = self.CSVFiles(
            self.csv_files,
            accept=AcceptCSVFile(self.dirs["root"] / "accept_results.csv"),
            changes=ChangesCSVFile(self.dirs["root"] / "changes_results.csv"),
            clean=CleanCSVFile(self.dirs["root"] / "clean_results.csv"),
            mip=MIPCSVFile(self.dirs["root"] / "mip_results.csv"),
            compromise=CompromiseCSVFile(self.dirs["root"] / "compromise_results.csv"),
            configs=ConfigCSVFile(self.dirs["root"] / "configs.csv"),
            group_parameters=GroupParametersCSVFile(
                self.dirs["root"] / "group_parameters.csv"
            ),
            path=PathCSVFile(self.dirs["root"] / "path_results.csv"),
        )

    def A(self, m: int, n: int, id: int):
        return self.dirs["A"] / filename_csv(locals())

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
    ):
        return self.dirs["D"] / filename_csv(locals())

    def Di(
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
        path: bool,
        P_id: int,
        it: int,
    ):
        return self.dirs["Di"] / filename_json(locals())

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
        path: bool,
        P_id: int,
        it: int,
    ):
        return self.dirs["Mc"] / filename_json(locals())

    def Dc(
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
        path: bool,
        P_id: int,
        it: int,
    ):
        return self.dirs["Dc"] / filename_csv(locals())

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
        Mc_id: int,
        path: bool,
        P_id: int,
        it: int,
    ):
        return self.dirs["C"] / filename_csv(locals())

    def Dr(
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
        path: bool,
        P_id: int,
        it: int,
    ):
        return self.dirs["Dr"] / filename_csv(locals())

    def Cr(
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
        path: bool,
        P_id: int,
        it: int,
    ):
        return self.dirs["Cr"] / filename_csv(locals())

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
        path: bool,
        id: int,
        it: int,
        t: int,
    ):
        return self.dirs["P"] / filename_csv(locals())
