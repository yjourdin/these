from ....utils import filename_csv, filename_json
from ...directory import Directory


class DirectoryGroupDecision(Directory):
    def __init__(self, dir: str, name: str):
        super().__init__(dir, name)
        self.dirs.update(
            A_train=self.dirs["root"] / "A_train",
            A_test=self.dirs["root"] / "A_test",
            Mo=self.dirs["root"] / "Mo",
            D=self.dirs["root"] / "D",
            Me=self.dirs["root"] / "Me",
        )

    def A_train(self, m: int, n: int, id: int):
        return self.dirs["A_train"] / filename_csv(locals())

    def A_test(self, m: int, n: int, id: int):
        return self.dirs["A_test"] / filename_csv(locals())

    def Mo(self, m: int, k: int, group_size: int, id: int):
        return self.dirs["Mo"] / filename_json(locals())

    def D(
        self,
        m: int,
        ntr: int,
        Atr_id: int,
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
