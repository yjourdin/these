from pathlib import Path
from typing import Literal

from model import ModelType


class Directory:
    def __init__(self, name: str):
        self.root_dir = Path(f"results/{name}/")
        self.A_train_dir = self.root_dir / "A_train"
        self.A_test_dir = self.root_dir / "A_test"
        self.Mo_dir = self.root_dir / "Mo"
        self.D_dir = self.root_dir / "D"
        self.Me_dir = self.root_dir / "Me"
        self.train_results_file = self.root_dir / "train_results.csv"
        self.test_results_file = self.root_dir / "test_results.csv"
        self.log_file = self.root_dir / "log.log"

    def A_train_file(self, i: int, n: int, m: int):
        return self.A_train_dir / f"DM_{i}_N_{n}_M_{m}.csv"

    def A_test_file(self, i: int, n: int, m: int):
        return self.A_test_dir / f"DM_{i}_N_{n}_M_{m}.csv"

    def Mo_file(self, i: int, m: int, model: ModelType, k: int):
        return self.Mo_dir / f"DM_{i}_M_{m}_model_{model}_K_{k}.json"

    def D_file(
        self, i: int, n_tr: int, m: int, Mo: ModelType, ko: int, n: int, e: float
    ):
        return self.D_dir / f"DM_{i}_Ntr_{n_tr}_M_{m}_Mo_{Mo}_Ko_{ko}_N_{n}_E_{e}.csv"

    def Me_file(
        self,
        i: int,
        n_tr: int,
        m: int,
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
            f"Ntr_{n_tr}_"
            f"M_{m}_"
            f"Mo_{Mo}_"
            f"Ko_{ko}_"
            f"N_{n}_"
            f"E_{e}_"
            f"Me_{Me}_"
            f"Ke_{ke}_"
            f"Method_{method}"
            + (f"_Config_{config}" if config is not None else "")
            + ".json"
        )

    def mkdir(self):
        self.root_dir.mkdir()
        self.A_train_dir.mkdir()
        self.A_test_dir.mkdir()
        self.Mo_dir.mkdir()
        self.D_dir.mkdir()
        self.Me_dir.mkdir()
        with self.train_results_file.open("w") as f:
            f.write("DM,M,Mo,Ko,N_tr,Error,Me,Ke,Method,Config,Time,Fitness,It.\n")
        with self.test_results_file.open("w") as f:
            f.write("DM,M,Mo,Ko,N_tr,Error,Me,Ke,Method,Config,Fitness,Kendall's tau\n")
