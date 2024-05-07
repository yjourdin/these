import csv
from pathlib import Path
from typing import Literal

from model import ModelType

RESULTS_DIR = Path("results")


class Directory:
    def __init__(self, name: str):
        self.root_dir = RESULTS_DIR / f"{name}"
        self.A_train_dir = self.root_dir / "A_train"
        self.A_test_dir = self.root_dir / "A_test"
        self.Mo_dir = self.root_dir / "Mo"
        self.D_dir = self.root_dir / "D"
        self.Me_dir = self.root_dir / "Me"
        self.train_results_file = self.root_dir / "train_results.csv"
        self.test_results_file = self.root_dir / "test_results.csv"
        self.log_file = self.root_dir / "log.log"
        self.seeds_file = self.root_dir / "seeds.csv"
        self.configs_file = self.root_dir / "configs.csv"

    def A_train_file(self, m: int, n: int, id: int):
        return self.A_train_dir / f"M_{m}_N_{n}_No_{id}.csv"

    def A_test_file(self, m: int, n: int, id: int):
        return self.A_test_dir / f"M_{m}_N_{n}_No_{id}.csv"

    def Mo_file(self, m: int, model: ModelType, k: int, id: int):
        return self.Mo_dir / f"M_{m}_model_{model}_K_{k}_No_{id}.json"

    def D_file(
        self,
        m: int,
        n_tr: int,
        A_tr_id: int,
        Mo: ModelType,
        ko: int,
        Mo_id: int,
        n: int,
        e: float,
    ):
        return self.D_dir / (
            f"M_{m}_"
            f"Ntr_{n_tr}_"
            f"AtrNo_{A_tr_id}_"
            f"Mo_{Mo}_"
            f"Ko_{ko}_"
            f"MoNo_{Mo_id}_"
            f"N_{n}_"
            f"E_{e}"
            ".csv"
        )

    def Me_file(
        self,
        m: int,
        n_tr: int,
        A_tr_id: int,
        Mo: ModelType,
        ko: int,
        Mo_id: int,
        n: int,
        e: float,
        Me: ModelType,
        ke: int,
        method: Literal["MIP", "SA"],
        config: int,
    ):
        return self.Me_dir / (
            f"M_{m}_"
            f"Ntr_{n_tr}_"
            f"AtrNo_{A_tr_id}_"
            f"Mo_{Mo}_"
            f"Ko_{ko}_"
            f"MoNo_{Mo_id}_"
            f"N_{n}_"
            f"E_{e}_"
            f"Me_{Me}_"
            f"Ke_{ke}_"
            f"Method_{method}_"
            f"Config_{config}"
            ".json"
        )

    def mkdir(self):
        self.root_dir.mkdir()
        self.A_train_dir.mkdir()
        self.A_test_dir.mkdir()
        self.Mo_dir.mkdir()
        self.D_dir.mkdir()
        self.Me_dir.mkdir()

        with self.seeds_file.open("w", newline="") as f:
            writer = csv.writer(f, "unix")
            writer.writerow(["Type", "Id", "Seed"])

        with self.configs_file.open("w", newline="") as f:
            writer = csv.writer(f, "unix")
            writer.writerow(["Method", "Id", "Config"])

        with self.train_results_file.open("w", newline="") as f:
            writer = csv.writer(f, "unix")
            writer.writerow(
                [
                    "DM",
                    "M",
                    "N_tr",
                    "Atr_id",
                    "Mo",
                    "Ko",
                    "Mo_id",
                    "N_bc",
                    "Error",
                    "Me",
                    "Ke",
                    "Method",
                    "Config",
                    "Time",
                    "Fitness",
                    "It.",
                ]
            )

        with self.test_results_file.open("w", newline="") as f:
            writer = csv.writer(f, "unix")
            writer.writerow(
                [
                    "DM",
                    "M",
                    "N_tr",
                    "Atr_id",
                    "Mo",
                    "Ko",
                    "Mo_id",
                    "N_bc",
                    "Error",
                    "Me",
                    "Ke",
                    "Method",
                    "Config",
                    "N_te",
                    "Ate_id",
                    "Fitness",
                    "Kendall's tau",
                ]
            )
