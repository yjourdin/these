from pathlib import Path
from typing import Literal

from src.methods import MethodEnum
from src.utils import filename_csv, filename_json, filename_log

from ...csv_file import CSVFile
from ...directory import Directory, DirectoryCSVFiles, DirectoryDirs
from ..elicitation.config import Config, MIPConfig
from ..elicitation.csv_files import ConfigCSVFile
from .csv_files import (
    AcceptCSVFile,
    ChangesCSVFile,
    CleanCSVFile,
    CollectiveCSVFile,
    CompromiseCSVFile,
    GroupParametersCSVFile,
    MieCSVFile,
    PathCSVFile,
)
from .fields import GroupParameters

DirectoryGroupDecisionDirs = (
    DirectoryDirs
    | Literal[
        "A",
        "Mo",
        "Mi",
        "D",
        "Di",
        "Mie",
        "Mcp",
        "MIP_log",
        "Mc",
        "Mp",
        "Dcp",
        "Dc",
        "C",
        "Da",
        "Cr",
        "Dr",
        "Dp",
        "P",
    ]
)

DirectoryGroupDecisionCSVFiles = (
    DirectoryCSVFiles
    | Literal[
        "accept",
        "changes",
        "clean",
        "mie",
        "collective",
        "compromise",
        "configs",
        "group_parameters",
        "path",
    ]
)


class DirectoryGroupDecision(Directory):
    def __init__(self, name: str, dir: Path | None = None):
        self.dirs: dict[DirectoryGroupDecisionDirs, Path]
        self.csv_files: dict[DirectoryGroupDecisionCSVFiles, CSVFile]
        super().__init__(name, dir)
        self.dirs |= {  # pyright: ignore[reportIncompatibleVariableOverride]
            "A": self.dirs["root"] / "A",
            "Mo": self.dirs["root"] / "Mo",
            "Mi": self.dirs["root"] / "Mi",
            "D": self.dirs["root"] / "D",
            "Di": self.dirs["root"] / "Di",
            "Mie": self.dirs["root"] / "Mie",
            "Mcp": self.dirs["root"] / "Mcp",
            "MIP_log": self.dirs["root"] / "MIP_log",
            "Mc": self.dirs["root"] / "Mc",
            "Mp": self.dirs["root"] / "Mp",
            "Dcp": self.dirs["root"] / "Dcp",
            "Dc": self.dirs["root"] / "Dc",
            "C": self.dirs["root"] / "C",
            "Da": self.dirs["root"] / "Da",
            "Cr": self.dirs["root"] / "Cr",
            "Dr": self.dirs["root"] / "Dr",
            "Dp": self.dirs["root"] / "Dp",
            "P": self.dirs["root"] / "P",
        }

        self.seeds = self.dirs["root"] / "seeds.json"

        self.csv_files |= {  # pyright: ignore[reportIncompatibleVariableOverride]
            "accept": AcceptCSVFile(self.dirs["root"] / "accept_results.csv"),
            "changes": ChangesCSVFile(self.dirs["root"] / "changes_results.csv"),
            "clean": CleanCSVFile(self.dirs["root"] / "clean_results.csv"),
            "mie": MieCSVFile(self.dirs["root"] / "mie_results.csv"),
            "collective": CollectiveCSVFile(
                self.dirs["root"] / "collective_results.csv"
            ),
            "compromise": CompromiseCSVFile(
                self.dirs["root"] / "compromise_results.csv"
            ),
            "configs": ConfigCSVFile(self.dirs["root"] / "configs.csv"),
            "group_parameters": GroupParametersCSVFile(
                self.dirs["root"] / "group_parameters.csv"
            ),
            "path": PathCSVFile(self.dirs["root"] / "path_results.csv"),
        }

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
        method: MethodEnum,
        config: Config,
        id: int,
        Mie: bool,
        Mie_config: MIPConfig | None,
        Mie_id: int,
        path: bool,
        P_id: int,
        it: int,
    ):
        return self.dirs["Di"] / filename_csv(locals())

    def Mie(
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
        dm_id: int,
    ):
        return self.dirs["Mie"] / filename_json(locals())

    def Mcp(
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
        method: MethodEnum,
        config: Config,
        Mie: bool,
        Mie_config: MIPConfig | None,
        Mie_id: int,
        Mc_id: int,
        id: int,
        path: bool,
        P_id: int,
        it: int,
    ):
        return self.dirs["Mcp"] / filename_json(locals())

    def MIP_log(
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
        method: MethodEnum,
        config: Config,
        Mie: bool,
        Mie_config: MIPConfig | None,
        Mie_id: int,
        Mc_id: int,
        id: int,
        path: bool,
        P_id: int,
        it: int,
    ):
        return self.dirs["MIP_log"] / filename_log(locals())

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
        method: MethodEnum,
        config: Config,
        Mie: bool,
        Mie_config: MIPConfig | None,
        Mie_id: int,
        id: int,
        path: bool,
        P_id: int,
        it: int,
    ):
        return self.dirs["Mc"] / filename_json(locals())

    def Dcp(
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
        method: MethodEnum,
        config: Config,
        Mie: bool,
        Mie_config: MIPConfig | None,
        Mie_id: int,
        Mc_id: int,
        id: int,
        path: bool,
        P_id: int,
        it: int,
    ):
        return self.dirs["Dcp"] / filename_csv(locals())

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
        method: MethodEnum,
        config: Config,
        id: int,
        Mie: bool,
        Mie_config: MIPConfig | None,
        Mie_id: int,
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
        method: MethodEnum,
        config: Config,
        Mc_id: int,
        Mie: bool,
        Mie_config: MIPConfig | None,
        Mie_id: int,
        path: bool,
        P_id: int,
        it: int,
    ):
        return self.dirs["C"] / filename_csv(locals())

    def Da(
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
        method: MethodEnum,
        config: Config,
        Mc_id: int,
        Mie: bool,
        Mie_config: MIPConfig | None,
        Mie_id: int,
        path: bool,
        P_id: int,
        it: int,
    ):
        return self.dirs["Da"] / filename_csv(locals())

    def Cr(
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
        method: MethodEnum,
        config: Config,
        Mc_id: int,
        Mie: bool,
        Mie_config: MIPConfig | None,
        Mie_id: int,
        path: bool,
        P_id: int,
        it: int,
    ):
        return self.dirs["Cr"] / filename_csv(locals())

    def Dr(
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
        method: MethodEnum,
        config: Config,
        Mc_id: int,
        Mie: bool,
        Mie_config: MIPConfig | None,
        Mie_id: int,
        path: bool,
        P_id: int,
        it: int,
    ):
        return self.dirs["Dr"] / filename_csv(locals())

    def Mp(
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
        method: MethodEnum,
        config: Config,
        Mc_id: int,
        Mie: bool,
        Mie_config: MIPConfig | None,
        Mie_id: int,
        path: bool,
        id: int,
        it: int,
        t: int,
    ):
        return self.dirs["Mp"] / filename_csv(locals())

    def Dp(
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
        method: MethodEnum,
        config: Config,
        Mc_id: int,
        Mie: bool,
        Mie_config: MIPConfig | None,
        Mie_id: int,
        path: bool,
        id: int,
        it: int,
        t: int,
    ):
        return self.dirs["Dp"] / filename_csv(locals())

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
        method: MethodEnum,
        config: Config,
        Mc_id: int,
        Mie: bool,
        Mie_config: MIPConfig | None,
        Mie_id: int,
        path: bool,
        id: int,
        it: int,
    ):
        return self.dirs["P"] / filename_csv(locals())
