from ..enum import StrEnum


class Fieldnames(StrEnum):
    pass


class TrainFieldnames(Fieldnames):
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


class TestFieldnames(Fieldnames):
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
    Mo_Intra_Kendall = "Mo Intra Kendall's tau"
    Me_Intra_Kendall = "Me Intra Kendall's tau"


class ConfigFieldnames(Fieldnames):
    Id = "Id"
    Method = "Method"
    Config = "Config"


class SeedFieldnames(Fieldnames):
    Task = "Task"
    Seed = "Seed"


class DSizeFieldnames(Fieldnames):
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
