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

    def A_train_file(self, i: int, m: int, n: int):
        return self.A_train_dir / f"DM_{i}_M_{m}_N_{n}.csv"

    def A_test_file(self, i: int, m: int, n: int):
        return self.A_test_dir / f"DM_{i}_M_{m}_N_{n}.csv"

    def Mo_file(self, i: int, m: int, model: ModelType, k: int):
        return self.Mo_dir / f"DM_{i}_M_{m}_model_{model}_K_{k}.json"

    def D_file(
        self, i: int, m: int, n_tr: int, Mo: ModelType, ko: int, n: int, e: float
    ):
        return self.D_dir / f"DM_{i}_M_{m}_Ntr_{n_tr}_Mo_{Mo}_Ko_{ko}_N_{n}_E_{e}.csv"

    def Me_file(
        self,
        i: int,
        m: int,
        n_tr: int,
        Mo: ModelType,
        ko: int,
        n: int,
        e: float,
        Me: ModelType,
        ke: int,
        method: Literal["MIP", "SA"],
        config: int | None = None,
    ):
        return self.Me_dir / (
            f"DM_{i}_"
            f"M_{m}_"
            f"Ntr_{n_tr}_"
            f"Mo_{Mo}_"
            f"Ko_{ko}_"
            f"N_{n}_"
            f"E_{e}_"
            f"Me_{Me}_"
            f"Ke_{ke}_"
            f"Method_{method}"
            f"_Config_{config}"
            ".json"
        )

    def mkdir(self):
        self.root_dir.mkdir()
        self.A_train_dir.mkdir()
        self.A_test_dir.mkdir()
        self.Mo_dir.mkdir()
        self.D_dir.mkdir()
        self.Me_dir.mkdir()
        with self.train_results_file.open("w") as f:
            f.write(
                "DM,"
                "M,"
                "N_tr,"
                "Mo,"
                "Ko,"
                "N_bc,"
                "Error,"
                "Me,"
                "Ke,"
                "Method,"
                "Config,"
                "Time,"
                "Fitness,"
                "It.\n"
            )
        with self.test_results_file.open("w") as f:
            f.write(
                "DM,"
                "M,"
                "N_tr,"
                "Mo,"
                "Ko,"
                "N_bc,"
                "Error,"
                "Me,"
                "Ke,"
                "Method,"
                "Config,"
                "N_te,"
                "Fitness,"
                "Kendall's tau\n"
            )
