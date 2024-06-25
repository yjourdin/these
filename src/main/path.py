import csv
from functools import partial
from pathlib import Path
from typing import Any, Literal

from ..model import ModelType

RESULTS_DIR = "results"

FIELDNAMES = {
    "train_results": [
        "M",
        "N_tr",
        "Atr_id",
        "Mo",
        "Ko",
        "Mo_id",
        "N_bc",
        "Error",
        "D_id",
        "Me",
        "Ke",
        "Method",
        "Config",
        "Me_id",
        "Time",
        "Fitness",
        "It.",
    ],
    "test_results": [
        "M",
        "N_tr",
        "Atr_id",
        "Mo",
        "Ko",
        "Mo_id",
        "N_bc",
        "Error",
        "D_id",
        "Me",
        "Ke",
        "Method",
        "Config",
        "Me_id",
        "N_te",
        "Ate_id",
        "Fitness",
        "Kendall's tau",
    ],
    "configs": ["Id", "Method", "Config"],
    "seeds": ["Task", "Seed"],
}


def filename(dct: dict[str, Any], ext: str):
    return (
        "_".join(
            k.title().replace("_", "") + "_" + str(v)
            for k, v in dct.items()
            if k != "self"
        )
        + "."
        + ext
    )


filename_csv = partial(filename, ext="csv")
filename_json = partial(filename, ext="json")


class Directory:
    def __init__(self, dir: str, name: str):
        self.root_dir = Path(dir, name)
        self.A_train_dir = self.root_dir / "A_train"
        self.A_test_dir = self.root_dir / "A_test"
        self.Mo_dir = self.root_dir / "Mo"
        self.D_dir = self.root_dir / "D"
        self.Me_dir = self.root_dir / "Me"
        self.train_results = self.root_dir / "train_results.csv"
        self.test_results = self.root_dir / "test_results.csv"
        self.log = self.root_dir / "log.log"
        self.args = self.root_dir / "args.json"
        self.seeds = self.root_dir / "seeds.csv"
        self.configs = self.root_dir / "configs.csv"

    def A_train(self, m: int, n: int, id: int):
        return self.A_train_dir / filename_csv(locals())

    def A_test(self, m: int, n: int, id: int):
        return self.A_test_dir / filename_csv(locals())

    def Mo(self, m: int, model: ModelType, k: int, id: int):
        return self.Mo_dir / filename_json(locals())

    def D(
        self,
        m: int,
        ntr: int,
        Atr_id: int,
        Mo: ModelType,
        ko: int,
        Mo_id: int,
        n: int,
        e: float,
        id: int,
    ):
        return self.D_dir / filename_csv(locals())

    def Me(
        self,
        m: int,
        ntr: int,
        Atr_id: int,
        Mo: ModelType,
        ko: int,
        Mo_id: int,
        n: int,
        e: float,
        D_id: int,
        Me: ModelType,
        ke: int,
        method: Literal["MIP", "SA"],
        config_id: int,
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

        with self.seeds.open("w", newline="") as f:
            writer = csv.DictWriter(f, FIELDNAMES["seeds"], dialect="unix")
            writer.writeheader()

        with self.configs.open("w", newline="") as f:
            writer = csv.DictWriter(f, FIELDNAMES["configs"], dialect="unix")
            writer.writeheader()

        with self.train_results.open("w", newline="") as f:
            writer = csv.DictWriter(f, FIELDNAMES["train_results"], dialect="unix")
            writer.writeheader()

        with self.test_results.open("w", newline="") as f:
            writer = csv.DictWriter(f, FIELDNAMES["test_results"], dialect="unix")
            writer.writeheader()
