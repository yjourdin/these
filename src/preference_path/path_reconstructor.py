from src.dataclass import Dataclass, dataclass, field


@dataclass(kw_only=True)
class Paths[T](Dataclass):
    parent: dict[T, dict[int, T | None]] = field(default_factory=dict)

    def paths(self, v: T) -> dict[int, list[T]]:
        result = {}
        parent = self.parent[v]
        for i, parent in self.parent[v].items():
            result |= {i: [v] + l for (i, l) in self.paths(parent).items()} if parent is not None else {i: [v]}
        return result
