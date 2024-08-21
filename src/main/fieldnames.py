from ..enum import StrEnum


class TrainFieldnames(StrEnum):
    M = "M"
    N_tr = "N_tr"
    Atr_id = "Atr_id"
    Mo = "Mo"
    Ko = "Ko"
    Group_size = "Group_size"
    Mo_id = "Mo_id"
    N_bc = "N_bc"
    Same_alt = "Same_alt"
    Error = "Error"
    D_id = "D_id"
    Me = "Me"
    Ke = "Ke"
    Method = "Method"
    Config = "Config"
    Me_id = "Me_id"
    Time = "Time"
    Fitness = "Fitness"
    It = "It."


class TestFieldnames(StrEnum):
    M = "M"
    N_tr = "N_tr"
    Atr_id = "Atr_id"
    Mo = "Mo"
    Ko = "Ko"
    Group_size = "Group_size"
    Mo_id = "Mo_id"
    N_bc = "N_bc"
    Same_alt = "Same_alt"
    Error = "Error"
    D_id = "D_id"
    Me = "Me"
    Ke = "Ke"
    Method = "Method"
    Config = "Config"
    Me_id = "Me_id"
    N_te = "N_te"
    Ate_id = "Ate_id"
    Fitness = "Fitness"
    Kendall = "Kendall's tau"


class ConfigFieldnames(StrEnum):
    Id = "Id"
    Method = "Method"
    Config = "Config"


class SeedFieldnames(StrEnum):
    Task = "Task"
    Seed = "Seed"


class DSizeFieldnames(StrEnum):
    M = "M"
    N_tr = "N_tr"
    Atr_id = "Atr_id"
    Mo = "Mo"
    Ko = "Ko"
    Group_size = "Group_size"
    Mo_id = "Mo_id"
    N_bc = "N_bc"
    Same_alt = "Same_alt"
    Error = "Error"
    D_id = "D_id"
    Size = "Size"


FIELDNAMES = {
    "train_results": [
        "M",
        "N_tr",
        "Atr_id",
        "Mo",
        "Ko",
        "Group_size",
        "Mo_id",
        "N_bc",
        "Same_alt",
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
        "Group_size",
        "Mo_id",
        "N_bc",
        "Same_alt",
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
    "D_size": ["Task", "Size"],
}
