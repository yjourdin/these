from pathlib import Path

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

    def A_train_file(self, i: int, m: int):
        return self.A_train_dir / f"No_{i}_M_{m}.csv"

    def A_test_file(self, i: int, m: int):
        return self.A_test_dir / f"No_{i}_M_{m}.csv"

    def Mo_file(self, i: int, m: int, model: ModelType, k: int):
        return self.Mo_dir / f"No_{i}_M_{m}_model_{model}_K_{k}.json"

    def D_file(self, i: int, m: int, Mo: ModelType, ko: int, n: int, e: float):
        return self.D_dir / f"No_{i}_M_{m}_Mo_{Mo}_Ko_{ko}_N_{n}_E_{e}.csv"

    def Me_file(
        self,
        i: int,
        Mo: ModelType,
        m: int,
        ko: int,
        n: int,
        e: float,
        Me: ModelType,
        ke: int,
    ):
        return (
            self.Me_dir
            / f"No_{i}_M_{m}_Mo_{Mo}_Ko_{ko}_N_{n}_E_{e}_Me_{Me}_Ke_{ke}.json"
        )

    def mkdir(self):
        self.root_dir.mkdir()
        self.A_train_dir.mkdir()
        self.A_test_dir.mkdir()
        self.Mo_dir.mkdir()
        self.D_dir.mkdir()
        self.Me_dir.mkdir()
        with self.train_results_file.open('w') as f:
            f.write("No.,M,Mo,Ko,N_tr,Error,Me,Ke,Time,It.,Fitness\n")
        with self.test_results_file.open('w') as f:
            f.write("No.,M,Mo,Ko,N_tr,Error,Me,Ke,Fitness,Kendall's tau\n")
