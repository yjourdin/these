from ..enum_base import StrEnum


class Fieldnames(StrEnum):
    pass


class TimeFieldnames(Fieldnames):
    Task = "Task"
    Time = "Time"
    

class SeedFieldnames(Fieldnames):
    Task = "Task"
    Seed = "Seed"
